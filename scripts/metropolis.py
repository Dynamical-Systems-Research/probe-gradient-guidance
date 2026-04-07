"""
Self-correcting search for Crystalite EDM sampler.

Ports the MatterGen MetropolisBandGapProposalScorer to work with
Crystalite's standard Transformer architecture. Key simplifications:
  - No GemNet hooks needed: directly extract Transformer hidden states
  - No scatter_mean: just index h[:, :-1, :] and mean-pool per structure
  - Same Metropolis energy function and accept/reject logic

Usage:
  from scripts.self_correction import BandGapScorer, edm_sampler_with_sc
"""
from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn as nn

from src.crystalite.crystalite import CrystaliteModel, mod1
from src.crystalite.edm_utils import karras_sigma_steps, denoise_edm, sigma_to_cnoise


def distance_to_interval(x: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    """Distance from x to [lo, hi]. Zero if inside."""
    return torch.clamp(lo - x, min=0.0) + torch.clamp(x - hi, min=0.0)


class BandGapProbe(nn.Module):
    """2-layer MLP probe trained on Crystalite Transformer hidden states."""

    def __init__(self, input_dim: int, hidden_dim: int = 256, include_timestep: bool = True):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.include_timestep = include_timestep
        effective_dim = input_dim + (1 if include_timestep else 0)
        self.backbone = nn.Sequential(
            nn.Linear(effective_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.band_gap_head = nn.Linear(hidden_dim, 1)
        self.window_head = nn.Linear(hidden_dim, 1)

    def forward(
        self, x: torch.Tensor, timestep: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        if self.include_timestep:
            if timestep is None:
                raise ValueError("BandGapProbe expects timestep features.")
            t = timestep.reshape(-1, 1).to(dtype=x.dtype, device=x.device)
            x = torch.cat([x, t], dim=-1)
        hidden = self.backbone(x)
        return {
            "band_gap": self.band_gap_head(hidden).squeeze(-1),
            "window_logit": self.window_head(hidden).squeeze(-1),
        }

    @classmethod
    def from_checkpoint(cls, path: str | Path) -> BandGapProbe:
        ckpt = torch.load(Path(path), map_location="cpu", weights_only=False)
        model = cls(
            input_dim=int(ckpt["input_dim"]),
            hidden_dim=int(ckpt.get("hidden_dim", 256)),
            include_timestep=bool(ckpt.get("include_timestep", True)),
        )
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        return model


def extract_atom_mean(
    model: CrystaliteModel,
    type_feats: torch.Tensor,
    frac_coords: torch.Tensor,
    lattice_feats: torch.Tensor,
    pad_mask: torch.Tensor,
    sigma: torch.Tensor,
    target_layer: int = -1,
) -> torch.Tensor:
    """
    Run Crystalite forward pass and extract mean-pooled atom embeddings
    from a specific Transformer layer.

    Args:
        target_layer: which layer to extract from (-1 = last layer = default)

    Returns:
        (B, d_model) mean-pooled atom embeddings
    """
    captured: list[torch.Tensor] = []

    # Hook the target layer
    blocks = model.trunk.blocks
    layer_idx = target_layer if target_layer >= 0 else len(blocks) + target_layer

    def hook_fn(module, input, output):
        captured.append(output.detach())

    handle = blocks[layer_idx].register_forward_hook(hook_fn)

    try:
        with torch.no_grad():
            t_sigma = sigma_to_cnoise(sigma)
            frac_mod = mod1(frac_coords)
            h_type = (model.type_proj(type_feats)
                      + model.coord_embed(frac_mod)
                      + model.segment_embed.weight[0])
            h_lat = model.lattice_embed(lattice_feats) + model.segment_embed.weight[1]
            h_lat = h_lat[:, None, :]
            x = torch.cat([h_type, h_lat], dim=1)
            pad_seq = torch.cat([
                pad_mask.bool(),
                torch.zeros((pad_mask.shape[0], 1), device=pad_mask.device, dtype=torch.bool),
            ], dim=1)
            t_emb = model.time(t_sigma, t_sigma)
            sigma_for_gem = torch.exp(4.0 * t_sigma.to(dtype=torch.float32))
            _ = model.trunk(x, t_emb, pad_mask=pad_seq, coords=frac_mod,
                            lattice=lattice_feats, t_sigma=sigma_for_gem)
    finally:
        handle.remove()

    if not captured:
        raise RuntimeError(f"Hook on layer {layer_idx} did not fire.")

    h = captured[0].float()  # (B, N+1, d_model)
    atom_h = h[:, :-1, :]  # (B, N, d_model)
    real_mask = ~pad_mask
    atom_h_masked = atom_h * real_mask.unsqueeze(-1).float()
    n_atoms = real_mask.float().sum(dim=1, keepdim=True).clamp_min(1.0)
    atom_mean = atom_h_masked.sum(dim=1) / n_atoms  # (B, d_model)
    return atom_mean


class BandGapScorer:
    """Metropolis accept/reject scorer for Crystalite EDM sampler.

    Scores proposals using a band-gap probe on Transformer hidden states.
    Supports hard constraints and best-of-K proposals.
    """

    def __init__(
        self,
        *,
        probe: BandGapProbe,
        lower_window: float = 4.0,
        upper_window: float = 6.0,
        target_band_gap: float = 5.0,
        max_scoring_t: float = 0.1,
        temperature: float = 0.1,
        interval_weight: float = 1.0,
        center_weight: float = 0.2,
        in_window_weight: float = 0.3,
        window_exit_penalty: float = 0.5,
        hard_floor: float | None = None,
        hard_ceiling: float | None = None,
        num_proposals: int = 1,
        probe_layer: int = -1,
    ):
        self.probe = probe
        self.lower_window = lower_window
        self.upper_window = upper_window
        self.target_band_gap = target_band_gap
        self.max_scoring_t = max_scoring_t
        self.temperature = temperature
        self.interval_weight = interval_weight
        self.center_weight = center_weight
        self.in_window_weight = in_window_weight
        self.window_exit_penalty = window_exit_penalty
        self.hard_floor = hard_floor
        self.hard_ceiling = hard_ceiling
        self.num_proposals = max(1, num_proposals)
        self.probe_layer = probe_layer

    def should_apply(self, sigma_val: float) -> bool:
        return sigma_val <= self.max_scoring_t

    def _energy(
        self,
        band_gap: torch.Tensor,
        window_logit: torch.Tensor,
        current_in_window: torch.Tensor | None = None,
    ) -> torch.Tensor:
        center_radius = max((self.upper_window - self.lower_window) / 2, 1e-6)
        interval_penalty = distance_to_interval(
            band_gap, self.lower_window, self.upper_window
        ) / center_radius
        center_penalty = torch.abs(band_gap - self.target_band_gap) / center_radius
        energy = self.interval_weight * interval_penalty + self.center_weight * center_penalty
        energy = energy - self.in_window_weight * torch.sigmoid(window_logit)

        if current_in_window is not None:
            proposed_in_window = interval_penalty <= 1e-8
            exits_window = current_in_window & ~proposed_in_window
            if bool(torch.any(exits_window)):
                energy = energy + exits_window.float() * self.window_exit_penalty

        return energy

    def _hard_constraint_mask(self, band_gap: torch.Tensor) -> torch.Tensor:
        mask = torch.ones_like(band_gap, dtype=torch.bool)
        if self.hard_floor is not None:
            mask = mask & (band_gap >= self.hard_floor)
        if self.hard_ceiling is not None:
            mask = mask & (band_gap <= self.hard_ceiling)
        return mask

    @torch.no_grad()
    def _predict(
        self,
        model: CrystaliteModel,
        type_feats: torch.Tensor,
        frac_coords: torch.Tensor,
        lattice_feats: torch.Tensor,
        pad_mask: torch.Tensor,
        sigma: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract atom_mean from Transformer and run probe."""
        atom_mean = extract_atom_mean(
            model, type_feats, frac_coords, lattice_feats,
            pad_mask, sigma, target_layer=self.probe_layer,
        )
        if next(self.probe.parameters()).device != atom_mean.device:
            self.probe = self.probe.to(atom_mean.device)
        outputs = self.probe(atom_mean, sigma)
        return outputs["band_gap"], outputs["window_logit"]

    @torch.no_grad()
    def accept_reject(
        self,
        model: CrystaliteModel,
        *,
        current_type: torch.Tensor,
        current_frac: torch.Tensor,
        current_lat: torch.Tensor,
        proposed_type: torch.Tensor,
        proposed_frac: torch.Tensor,
        proposed_lat: torch.Tensor,
        pad_mask: torch.Tensor,
        sigma: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Metropolis accept/reject with hard constraints.

        Returns:
            (accepted_type, accepted_frac, accepted_lat) — per-sample
            accepted state (current or proposed).
        """
        cur_bg, cur_wl = self._predict(
            model, current_type, current_frac, current_lat, pad_mask, sigma
        )
        prop_bg, prop_wl = self._predict(
            model, proposed_type, proposed_frac, proposed_lat, pad_mask, sigma
        )

        hard_ok = self._hard_constraint_mask(prop_bg)

        current_in_window = distance_to_interval(
            cur_bg, self.lower_window, self.upper_window
        ) <= 1e-8

        cur_energy = self._energy(cur_bg, cur_wl)
        prop_energy = self._energy(prop_bg, prop_wl, current_in_window=current_in_window)

        delta = prop_energy - cur_energy
        accept = delta <= 0
        uphill = ~accept
        if bool(torch.any(uphill)):
            probs = torch.exp(-delta[uphill] / self.temperature).clamp(max=1.0)
            accept[uphill] = torch.rand_like(probs) < probs

        accept = accept & hard_ok

        sel = accept.unsqueeze(-1)  # (B, 1)
        out_type = torch.where(sel.unsqueeze(-1), proposed_type, current_type)
        out_frac = torch.where(sel.unsqueeze(-1), proposed_frac, current_frac)
        out_lat = torch.where(sel, proposed_lat, current_lat)

        return out_type, out_frac, out_lat


@torch.no_grad()
def edm_sampler_with_sc(
    model: CrystaliteModel,
    pad_mask: torch.Tensor,
    type_dim: int,
    num_steps: int,
    sigma_min: float,
    sigma_max: float,
    rho: float,
    S_churn: float,
    S_min: float,
    S_max: float,
    S_noise: float,
    sigma_data_type: float,
    sigma_data_coord: float,
    sigma_data_lat: float,
    scorer: BandGapScorer | None = None,
    generator: torch.Generator | None = None,
    autocast_dtype: torch.dtype | None = None,
    fixed_atom_types: torch.Tensor | None = None,
    skip_type_scaling: bool = False,
    lattice_repr: str = "y1",
) -> dict[str, torch.Tensor]:
    """
    EDM Heun sampler with optional self-correcting search.

    Based on Crystalite's edm_sampler but adds Metropolis accept/reject
    at each step where scorer.should_apply(sigma) is True.
    """
    from src.crystalite.sampler import wrap_frac, clamp_lattice_latent

    device = pad_mask.device
    bsz, nmax = pad_mask.shape
    real_mask = ~pad_mask

    type_x = torch.randn((bsz, nmax, type_dim), device=device, generator=generator)
    frac_x = torch.randn((bsz, nmax, 3), device=device, generator=generator)
    lat_x = torch.randn((bsz, 6), device=device, generator=generator)

    type_x = torch.where(real_mask[..., None], type_x, torch.zeros_like(type_x))
    frac_x = torch.where(real_mask[..., None], frac_x, torch.zeros_like(frac_x))

    t_steps = karras_sigma_steps(
        num_steps=num_steps, sigma_min=sigma_min, sigma_max=sigma_max,
        rho=rho, device=device,
    )

    type_next = type_x * t_steps[0]
    frac_next = frac_x * t_steps[0]
    lat_next = lat_x * t_steps[0]

    if fixed_atom_types is not None:
        type_next = fixed_atom_types.to(dtype=type_next.dtype)

    sc_applied = 0

    for i, (t_cur, t_next_val) in enumerate(zip(t_steps[:-1], t_steps[1:])):
        gamma = (
            min(S_churn / num_steps, math.sqrt(2.0) - 1.0)
            if (t_cur >= S_min and t_cur <= S_max)
            else 0.0
        )
        t_hat = t_cur + gamma * t_cur
        noise_scale = (t_hat**2 - t_cur**2).sqrt()

        type_hat = type_next + noise_scale * S_noise * torch.randn_like(type_next, generator=generator)
        frac_hat = frac_next + noise_scale * S_noise * torch.randn_like(frac_next, generator=generator)
        lat_hat = lat_next + noise_scale * S_noise * torch.randn_like(lat_next, generator=generator)

        if fixed_atom_types is not None:
            type_hat = fixed_atom_types.to(dtype=type_hat.dtype)

        sigma_hat = torch.full((bsz,), float(t_hat), device=device)
        denoised = denoise_edm(
            model=model,
            type_noisy=type_hat, frac_noisy=frac_hat, lat_noisy=lat_hat,
            pad_mask=pad_mask, sigma=sigma_hat,
            sigma_data_type=sigma_data_type, sigma_data_coord=sigma_data_coord,
            sigma_data_lat=sigma_data_lat, sigma_min=sigma_min, sigma_max=sigma_max,
            autocast_dtype=autocast_dtype, skip_type_scaling=skip_type_scaling,
        )

        type_dx = (type_hat - denoised["type"]) / t_hat
        frac_dx = wrap_frac(frac_hat - denoised["frac"]) / t_hat
        lat_dx = (lat_hat - denoised["lat"]) / t_hat

        type_next = type_hat + (t_next_val - t_hat) * type_dx
        if fixed_atom_types is not None:
            type_next = fixed_atom_types.to(dtype=type_next.dtype)
        frac_next = frac_hat + (t_next_val - t_hat) * frac_dx
        lat_next = lat_hat + (t_next_val - t_hat) * lat_dx
        frac_next = frac_next - torch.round(frac_next)

        # Heun second-order correction
        if i < num_steps - 1:
            sigma_next = torch.full((bsz,), float(t_next_val), device=device)
            denoised_next = denoise_edm(
                model=model,
                type_noisy=type_next, frac_noisy=frac_next, lat_noisy=lat_next,
                pad_mask=pad_mask, sigma=sigma_next,
                sigma_data_type=sigma_data_type, sigma_data_coord=sigma_data_coord,
                sigma_data_lat=sigma_data_lat, sigma_min=sigma_min, sigma_max=sigma_max,
                autocast_dtype=autocast_dtype, skip_type_scaling=skip_type_scaling,
            )

            type_dx2 = (type_next - denoised_next["type"]) / t_next_val
            frac_dx2 = wrap_frac(frac_next - denoised_next["frac"]) / t_next_val
            lat_dx2 = (lat_next - denoised_next["lat"]) / t_next_val

            # Save pre-correction state for Metropolis
            type_pre = type_next.clone()
            frac_pre = frac_next.clone()
            lat_pre = lat_next.clone()

            type_next = type_hat + (t_next_val - t_hat) * (0.5 * type_dx + 0.5 * type_dx2)
            if fixed_atom_types is not None:
                type_next = fixed_atom_types.to(dtype=type_next.dtype)
            frac_next = frac_hat + (t_next_val - t_hat) * (0.5 * frac_dx + 0.5 * frac_dx2)
            lat_next = lat_hat + (t_next_val - t_hat) * (0.5 * lat_dx + 0.5 * lat_dx2)
            frac_next = frac_next - torch.round(frac_next)

            # Self-correction: Metropolis accept/reject on the Heun correction
            if scorer is not None and scorer.should_apply(float(t_next_val)):
                sc_applied += 1
                type_next, frac_next, lat_next = scorer.accept_reject(
                    model,
                    current_type=type_pre, current_frac=frac_pre, current_lat=lat_pre,
                    proposed_type=type_next, proposed_frac=frac_next, proposed_lat=lat_next,
                    pad_mask=pad_mask, sigma=sigma_next,
                )

    type_next = torch.where(real_mask[..., None], type_next, torch.zeros_like(type_next))
    frac_next = torch.where(real_mask[..., None], frac_next, torch.zeros_like(frac_next))
    lat_next = torch.nan_to_num(lat_next, nan=0.0, posinf=10.0, neginf=-10.0)
    type_next = torch.nan_to_num(type_next, nan=0.0, posinf=10.0, neginf=-10.0)
    frac_next = torch.nan_to_num(frac_next, nan=0.0, posinf=1.0, neginf=0.0)
    lat_next = clamp_lattice_latent(lat_next, lattice_repr=lattice_repr)
    type_next = type_next.clamp(min=-50.0, max=50.0)

    return {"type": type_next, "frac": frac_next, "lat": lat_next}
