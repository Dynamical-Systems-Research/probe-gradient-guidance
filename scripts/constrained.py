"""
Experiment 7: Hybrid constrained generation.

Gradient guidance for continuous properties (band gap steering).
Token masking for discrete constraints (element exclusion/inclusion).

These are applied at different stages:
- Gradient guidance: during denoising (modifies trajectory)
- Token masking: at decode time (modifies final atom type selection)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.crystalite.sampler import edm_sampler
from src.models.type_encoding import build_type_encoding
from scripts.self_correction import BandGapProbe, extract_atom_mean
from scripts.train_probe import load_model
from scripts.guided_sampler import guided_edm_sampler

REFRACTORY_Z = {74, 42, 73, 41, 23, 24, 40, 72, 75}  # W, Mo, Ta, Nb, V, Cr, Zr, Hf, Re
EXCLUDED_Z = {27, 28}  # Co, Ni
ALL_ELEMENTS = set(range(1, 90))


def masked_decode(type_logits: torch.Tensor, pad_mask: torch.Tensor,
                  type_enc, exclude_z: set | None = None,
                  boost_z: set | None = None, boost_strength: float = 5.0):
    """
    Decode type logits to atomic numbers with compositional constraints.

    exclude_z: elements to hard-exclude (set logits to -inf)
    boost_z: elements to boost (add boost_strength to logits)
    """

    vz = 89
    allowed = torch.ones(vz, dtype=torch.bool)
    if exclude_z:
        for z in exclude_z:
            if 1 <= z <= vz:
                allowed[z - 1] = False

    dummy_a0 = torch.arange(1, vz + 1).unsqueeze(0)  # (1, 89)
    dummy_pad = torch.zeros(1, vz, dtype=torch.bool)
    codebook = type_enc.encode_from_A0(dummy_a0, dummy_pad).squeeze(0)  # (89, type_dim)
    codebook = codebook.to(type_logits.device)

    # type_logits: (B, N, d), codebook: (89, d)
    logits_norm = torch.nn.functional.normalize(type_logits.float(), dim=-1)
    codebook_norm = torch.nn.functional.normalize(codebook.float(), dim=-1)
    similarity = torch.einsum("bnd,ed->bne", logits_norm, codebook_norm)  # (B, N, 89)

    if exclude_z:
        similarity[:, :, ~allowed] = -1e9

    if boost_z:
        for z in boost_z:
            if 1 <= z <= vz:
                similarity[:, :, z - 1] += boost_strength

    decoded = similarity.argmax(dim=-1) + 1  # (B, N), elements 1-89
    decoded = torch.where(~pad_mask, decoded, torch.zeros_like(decoded))
    return decoded


def evaluate_structures(decoded: torch.Tensor, pad_mask: torch.Tensor,
                        bg_values: np.ndarray | None = None):
    """Compute composition and property metrics."""
    real_mask = ~pad_mask
    results = {
        "n_total": decoded.shape[0],
        "ref_fracs": [], "has_excluded": [], "compositions": [],
        "unique_elements": set(),
    }

    for i in range(decoded.shape[0]):
        atoms = decoded[i][real_mask[i]].cpu().tolist()
        n = len(atoms)
        ref_count = sum(1 for z in atoms if z in REFRACTORY_Z)
        has_excl = any(z in EXCLUDED_Z for z in atoms)
        results["ref_fracs"].append(ref_count / max(n, 1))
        results["has_excluded"].append(has_excl)
        results["compositions"].append(atoms)
        results["unique_elements"].update(atoms)

    ref_fracs = np.array(results["ref_fracs"])
    has_excl = np.array(results["has_excluded"])

    summary = {
        "n_total": results["n_total"],
        "ref_frac_mean": float(ref_fracs.mean()),
        "ref_frac_nonzero": float((ref_fracs > 0).mean()),
        "excluded_rate": float(has_excl.mean()),
        "n_unique_elements": len(results["unique_elements"]),
        "sample_compositions": results["compositions"][:6],
    }

    if bg_values is not None:
        bg = np.array(bg_values)
        summary["mean_bg"] = float(bg.mean())
        summary["insulator_rate"] = float((bg > 0.5).mean())
        summary["in_window_rate"] = float(((bg >= 4.0) & (bg <= 6.0)).mean())

    return summary


def score_bandgap(model, probe, result, pad_mask, device="cuda"):
    sigma = torch.full((result["type"].shape[0],), 0.01, device=device)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        am = extract_atom_mean(model, result["type"], result["frac"] + 0.5,
                               result["lat"], pad_mask, sigma, -1)
        out = probe(am.to(device), sigma)
    return out["band_gap"].float().cpu().numpy()


def main():
    device = "cuda"
    model = load_model("outputs/dng_alex_mp20/checkpoints/final.pt", device)
    probe = BandGapProbe.from_checkpoint("results/self_correction/probe.pt").to(device)
    type_enc = build_type_encoding("subatomic_tokenizer_pca_24", 89)

    N = 128
    nmax = 20

    sampler_base = dict(
        pad_mask=None, type_dim=type_enc.type_dim, num_steps=100,
        sigma_min=0.002, sigma_max=80.0, rho=7.0,
        S_churn=20.0, S_min=0.0, S_max=999.0, S_noise=1.0,
        sigma_data_type=1.0, sigma_data_coord=0.25, sigma_data_lat=1.0,
        autocast_dtype=torch.bfloat16, lattice_repr="y1",
    )

    configs = [
        # (label, guidance_weight, guidance_start_sigma, exclude_z, boost_z, boost_strength)
        ("baseline", 0, 0, None, None, 0),
        ("bandgap_only_w10", 10.0, 5.0, None, None, 0),
        ("exclude_CoNi", 0, 0, EXCLUDED_Z, None, 0),
        ("boost_refractory", 0, 0, None, REFRACTORY_Z, 3.0),
        ("hybrid_w10_exclude_CoNi", 10.0, 5.0, EXCLUDED_Z, None, 0),
        ("hybrid_w10_boost_ref", 10.0, 5.0, None, REFRACTORY_Z, 3.0),
        ("hybrid_w10_boost_ref_exclude_CoNi", 10.0, 5.0, EXCLUDED_Z, REFRACTORY_Z, 3.0),
        ("hybrid_w5_boost_ref5_exclude_CoNi", 5.0, 5.0, EXCLUDED_Z, REFRACTORY_Z, 5.0),
        ("hybrid_w10_boost_ref5_exclude_CoNi", 10.0, 5.0, EXCLUDED_Z, REFRACTORY_Z, 5.0),
    ]

    print(f"Generating {N} structures per config, 2 seeds")
    print(f"{'Config':>45} {'Seed':>4} {'RefFrac':>7} {'Ref%':>5} {'CoNi%':>5} "
          f"{'Ins%':>5} {'IW%':>5} {'MeanBG':>7}")
    print("-" * 100)

    all_results = []
    for seed in [42, 123]:
        for label, gw, gs, excl, boost, bs in configs:
            torch.manual_seed(seed)
            n_atoms = torch.randint(4, 13, (N,))
            pad_mask = (torch.arange(nmax).unsqueeze(0).expand(N, -1)
                        >= n_atoms.unsqueeze(1)).to(device)
            sampler_base["pad_mask"] = pad_mask

            if gw > 0:
                result = guided_edm_sampler(
                    model=model, probe=probe,
                    guidance_weight=gw, guidance_start_sigma=gs,
                    **sampler_base,
                )
            else:
                result = edm_sampler(model=model, **sampler_base)

            decoded = masked_decode(result["type"], pad_mask, type_enc,
                                    exclude_z=excl, boost_z=boost, boost_strength=bs)

            bg = score_bandgap(model, probe, result, pad_mask)

            metrics = evaluate_structures(decoded, pad_mask, bg)

            name = f"{label}_s{seed}"
            print(f"{name:>45} {seed:>4} {metrics['ref_frac_mean']:>7.3f} "
                  f"{metrics['ref_frac_nonzero']*100:>4.0f}% "
                  f"{metrics['excluded_rate']*100:>4.0f}% "
                  f"{metrics['insulator_rate']*100:>4.0f}% "
                  f"{metrics['in_window_rate']*100:>4.0f}% "
                  f"{metrics['mean_bg']:>7.2f}")

            if "hybrid" in label and seed == 42:
                from pymatgen.core.periodic_table import Element
                for j, comp in enumerate(metrics["sample_compositions"][:3]):
                    symbols = [Element.from_Z(z).symbol for z in comp]
                    ref_in = [s for s in symbols if Element(s).Z in REFRACTORY_Z]
                    print(f"{'':>49} sample {j}: {' '.join(symbols)} (ref: {ref_in})")

            all_results.append({"name": name, "seed": seed, "config": label, **metrics})

    print(f"\n{'='*100}")
    print("AVERAGED ACROSS SEEDS")
    print(f"{'='*100}")
    from collections import defaultdict
    grouped = defaultdict(list)
    for r in all_results:
        grouped[r["config"]].append(r)

    print(f"{'Config':>45} {'RefFrac':>7} {'Ref%':>5} {'CoNi%':>5} {'Ins%':>5} {'IW%':>5} {'MeanBG':>7}")
    print("-" * 85)
    for label, runs in grouped.items():
        rf = np.mean([r["ref_frac_mean"] for r in runs])
        rn = np.mean([r["ref_frac_nonzero"] for r in runs])
        ex = np.mean([r["excluded_rate"] for r in runs])
        ins = np.mean([r.get("insulator_rate", 0) for r in runs])
        iw = np.mean([r.get("in_window_rate", 0) for r in runs])
        mbg = np.mean([r.get("mean_bg", 0) for r in runs])
        print(f"{label:>45} {rf:>7.3f} {rn*100:>4.0f}% {ex*100:>4.0f}% "
              f"{ins*100:>4.0f}% {iw*100:>4.0f}% {mbg:>7.2f}")


if __name__ == "__main__":
    main()
