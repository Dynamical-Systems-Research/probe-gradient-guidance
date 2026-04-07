"""
Probe-gradient guided EDM sampler for Crystalite.

Classifier guidance analog: uses ∇_x log p(band_gap_in_window | x)
from the trained probe to actively steer the denoising trajectory
toward the target band-gap window.

This replaces Metropolis accept/reject (passive selection) with
active gradient-based steering — the probe does ALL the work.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.crystalite.crystalite import CrystaliteModel, mod1
from src.crystalite.edm_utils import karras_sigma_steps, denoise_edm, sigma_to_cnoise
from src.crystalite.sampler import wrap_frac, clamp_lattice_latent
from scripts.self_correction import BandGapProbe


def compute_probe_guidance(
    model: CrystaliteModel,
    probe: BandGapProbe,
    type_feats: torch.Tensor,
    frac_coords: torch.Tensor,
    lattice_feats: torch.Tensor,
    pad_mask: torch.Tensor,
    sigma: torch.Tensor,
    target_layer: int = -1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """
    Compute probe gradient w.r.t. model inputs for classifier guidance.

    Returns gradients for (type_feats, frac_coords, lattice_feats) and
    the current window_logit value for monitoring.
    """
    type_feats = type_feats.detach().requires_grad_(True)
    frac_coords = frac_coords.detach().requires_grad_(True)
    lattice_feats = lattice_feats.detach().requires_grad_(True)

    # Forward pass through Transformer trunk WITHOUT GEM (which has in-place ops).
    # GEM adds geometric attention bias but its _sanitize_lattice_latent uses
    # in-place slice assignment that breaks autograd. The gradient direction from
    # the probe is still valid without GEM — it just won't account for geometry bias.
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
    if pad_seq.dtype != torch.bool:
        pad_seq = pad_seq.bool()
    for block in model.trunk.blocks:
        if pad_seq is not None:
            x = x.masked_fill(pad_seq[..., None], 0.0)
        x = block(x, t_emb, pad_mask=pad_seq, attn_head_bias=None)
    h = model.trunk.norm_out(x)
    if pad_seq is not None:
        h = h.masked_fill(pad_seq[..., None], 0.0)

    h_float = h.float()
    atom_h = h_float[:, :-1, :]
    real_mask = ~pad_mask
    atom_h_masked = atom_h * real_mask.unsqueeze(-1).float()
    n_atoms = real_mask.float().sum(dim=1, keepdim=True).clamp_min(1.0)
    atom_mean = atom_h_masked.sum(dim=1) / n_atoms

    outputs = probe(atom_mean, sigma)
    window_logit = outputs["window_logit"]

    # Maximize window_logit = maximize P(in_window | x)
    # Gradient of log-sigmoid(window_logit) w.r.t. inputs
    log_prob = torch.nn.functional.logsigmoid(window_logit).sum()
    log_prob.backward()

    grad_type = type_feats.grad.detach() if type_feats.grad is not None else torch.zeros_like(type_feats)
    grad_frac = frac_coords.grad.detach() if frac_coords.grad is not None else torch.zeros_like(frac_coords)
    grad_lat = lattice_feats.grad.detach() if lattice_feats.grad is not None else torch.zeros_like(lattice_feats)

    window_val = float(torch.sigmoid(window_logit).mean().item())

    return grad_type, grad_frac, grad_lat, window_val


@torch.no_grad()
def guided_edm_sampler(
    model: CrystaliteModel,
    probe: BandGapProbe,
    pad_mask: torch.Tensor,
    type_dim: int,
    num_steps: int,
    guidance_weight: float = 1.0,
    guidance_start_sigma: float = 5.0,
    sigma_min: float = 0.002,
    sigma_max: float = 80.0,
    rho: float = 7.0,
    S_churn: float = 20.0,
    S_min: float = 0.0,
    S_max: float = 999.0,
    S_noise: float = 1.0,
    sigma_data_type: float = 1.0,
    sigma_data_coord: float = 0.25,
    sigma_data_lat: float = 1.0,
    autocast_dtype: torch.dtype | None = torch.bfloat16,
    lattice_repr: str = "y1",
    generator: torch.Generator | None = None,
) -> dict[str, torch.Tensor]:
    """
    EDM Heun sampler with probe-gradient guidance.

    At each step where sigma <= guidance_start_sigma:
    1. Compute standard denoised output
    2. Compute ∇_x log p(in_window | x) from the probe
    3. Add guidance_weight * sigma * gradient to the denoised output
    """
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

    guidance_applied = 0
    window_history = []

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

        sigma_hat = torch.full((bsz,), float(t_hat), device=device)

        denoised = denoise_edm(
            model=model,
            type_noisy=type_hat, frac_noisy=frac_hat, lat_noisy=lat_hat,
            pad_mask=pad_mask, sigma=sigma_hat,
            sigma_data_type=sigma_data_type, sigma_data_coord=sigma_data_coord,
            sigma_data_lat=sigma_data_lat, sigma_min=sigma_min, sigma_max=sigma_max,
            autocast_dtype=autocast_dtype,
        )

        type_d = denoised["type"]
        frac_d = denoised["frac"]
        lat_d = denoised["lat"]

        if float(t_hat) <= guidance_start_sigma and guidance_weight > 0:
            with torch.enable_grad():
                grad_type, grad_frac, grad_lat, window_val = compute_probe_guidance(
                    model, probe,
                    type_d, frac_d + 0.5, lat_d,
                    pad_mask, sigma_hat,
                )

            # Classifier guidance: shift denoised toward higher P(in_window)
            # Scale by sigma to match EDM noise schedule
            scale = float(t_hat) * guidance_weight
            type_d = type_d + scale * grad_type
            frac_d = frac_d + scale * grad_frac
            lat_d = lat_d + scale * grad_lat

            type_d = type_d * real_mask.unsqueeze(-1).float()
            frac_d = frac_d * real_mask.unsqueeze(-1).float()

            guidance_applied += 1
            window_history.append(window_val)

        type_dx = (type_hat - type_d) / t_hat
        frac_dx = wrap_frac(frac_hat - frac_d) / t_hat
        lat_dx = (lat_hat - lat_d) / t_hat

        type_next = type_hat + (t_next_val - t_hat) * type_dx
        frac_next = frac_hat + (t_next_val - t_hat) * frac_dx
        lat_next = lat_hat + (t_next_val - t_hat) * lat_dx
        frac_next = frac_next - torch.round(frac_next)

        if i < num_steps - 1:
            sigma_next = torch.full((bsz,), float(t_next_val), device=device)
            denoised_next = denoise_edm(
                model=model,
                type_noisy=type_next, frac_noisy=frac_next, lat_noisy=lat_next,
                pad_mask=pad_mask, sigma=sigma_next,
                sigma_data_type=sigma_data_type, sigma_data_coord=sigma_data_coord,
                sigma_data_lat=sigma_data_lat, sigma_min=sigma_min, sigma_max=sigma_max,
                autocast_dtype=autocast_dtype,
            )

            type_d2 = denoised_next["type"]
            frac_d2 = denoised_next["frac"]
            lat_d2 = denoised_next["lat"]

            if float(t_next_val) <= guidance_start_sigma and guidance_weight > 0:
                with torch.enable_grad():
                    g_type2, g_frac2, g_lat2, _ = compute_probe_guidance(
                        model, probe,
                        type_d2, frac_d2 + 0.5, lat_d2,
                        pad_mask, sigma_next,
                    )
                scale2 = float(t_next_val) * guidance_weight
                type_d2 = type_d2 + scale2 * g_type2
                frac_d2 = frac_d2 + scale2 * g_frac2
                lat_d2 = lat_d2 + scale2 * g_lat2
                type_d2 = type_d2 * real_mask.unsqueeze(-1).float()
                frac_d2 = frac_d2 * real_mask.unsqueeze(-1).float()

            type_dx2 = (type_next - type_d2) / t_next_val
            frac_dx2 = wrap_frac(frac_next - frac_d2) / t_next_val
            lat_dx2 = (lat_next - lat_d2) / t_next_val

            type_next = type_hat + (t_next_val - t_hat) * (0.5 * type_dx + 0.5 * type_dx2)
            frac_next = frac_hat + (t_next_val - t_hat) * (0.5 * frac_dx + 0.5 * frac_dx2)
            lat_next = lat_hat + (t_next_val - t_hat) * (0.5 * lat_dx + 0.5 * lat_dx2)
            frac_next = frac_next - torch.round(frac_next)

    type_next = torch.where(real_mask[..., None], type_next, torch.zeros_like(type_next))
    frac_next = torch.where(real_mask[..., None], frac_next, torch.zeros_like(frac_next))
    lat_next = torch.nan_to_num(lat_next, nan=0.0, posinf=10.0, neginf=-10.0)
    type_next = torch.nan_to_num(type_next, nan=0.0, posinf=10.0, neginf=-10.0)
    frac_next = torch.nan_to_num(frac_next, nan=0.0, posinf=1.0, neginf=0.0)
    lat_next = clamp_lattice_latent(lat_next, lattice_repr=lattice_repr)
    type_next = type_next.clamp(min=-50.0, max=50.0)

    return {
        "type": type_next, "frac": frac_next, "lat": lat_next,
        "guidance_applied": guidance_applied,
        "window_history": window_history,
    }
