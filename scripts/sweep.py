"""Sweep guidance weights at N=128 with 2 seeds."""
from __future__ import annotations
import sys
from pathlib import Path
import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.crystalite.sampler import edm_sampler
from src.models.type_encoding import build_type_encoding
from scripts.self_correction import BandGapProbe, extract_atom_mean
from scripts.train_probe import load_model
from scripts.guided_sampler import guided_edm_sampler

device = "cuda"
model = load_model("outputs/dng_alex_mp20/checkpoints/final.pt", device)
probe = BandGapProbe.from_checkpoint("results/self_correction/probe.pt").to(device)
type_enc = build_type_encoding("subatomic_tokenizer_pca_24", 89)


def score(result, pad_mask):
    sigma = torch.full((result["type"].shape[0],), 0.01, device=device)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        am = extract_atom_mean(model, result["type"], result["frac"] + 0.5,
                               result["lat"], pad_mask, sigma, -1)
        out = probe(am.to(device), sigma)
    return out["band_gap"].float().cpu().numpy()


N = 128
nmax = 20

configs = [
    ("baseline", 0.0, 0.0),
    ("guided_w1.0_s5", 1.0, 5.0),
    ("guided_w2.0_s5", 2.0, 5.0),
    ("guided_w3.0_s5", 3.0, 5.0),
    ("guided_w3.0_s1", 3.0, 1.0),
    ("guided_w5.0_s5", 5.0, 5.0),
    ("guided_w5.0_s1", 5.0, 1.0),
    ("guided_w10_s5", 10.0, 5.0),
]

print(f"Generating {N} structures per config, 2 seeds")
header = f"{'Config':>25} {'Seed':>5} {'MeanBG':>7} {'Metal%':>7} {'IW%':>6} {'IW#':>5}"
print(header)
print("-" * len(header))

results = []
for seed in [42, 123]:
    for label, w, gs in configs:
        torch.manual_seed(seed)
        n_atoms = torch.randint(4, nmax + 1, (N,))
        pad_mask = (torch.arange(nmax).unsqueeze(0).expand(N, -1)
                    >= n_atoms.unsqueeze(1)).to(device)

        sampler_kwargs = dict(
            pad_mask=pad_mask, type_dim=type_enc.type_dim, num_steps=100,
            sigma_min=0.002, sigma_max=80.0, rho=7.0,
            S_churn=20.0, S_min=0.0, S_max=999.0, S_noise=1.0,
            sigma_data_type=1.0, sigma_data_coord=0.25, sigma_data_lat=1.0,
            autocast_dtype=torch.bfloat16, lattice_repr="y1",
        )

        if w == 0:
            result = edm_sampler(model=model, **sampler_kwargs)
        else:
            result = guided_edm_sampler(
                model=model, probe=probe,
                guidance_weight=w, guidance_start_sigma=gs,
                **sampler_kwargs,
            )

        bg = score(result, pad_mask)
        metals = (bg < 0.5).mean() * 100
        iw_mask = (bg >= 4.0) & (bg <= 6.0)
        iw_pct = iw_mask.mean() * 100
        iw_count = int(iw_mask.sum())
        mean_bg = float(bg.mean())

        print(f"{label:>25} {seed:>5} {mean_bg:>7.2f} {metals:>6.1f}% {iw_pct:>5.1f}% {iw_count:>5}")
        results.append({
            "config": label, "seed": seed, "weight": w,
            "guidance_start": gs, "mean_bg": mean_bg,
            "metal_rate": metals / 100, "iw_rate": iw_pct / 100,
            "iw_count": iw_count, "n_total": N,
        })

print("\n" + "=" * 60)
print("AVERAGED ACROSS SEEDS")
print("=" * 60)
from collections import defaultdict
grouped = defaultdict(list)
for r in results:
    grouped[r["config"]].append(r)

print(f"{'Config':>25} {'MeanBG':>7} {'Metal%':>7} {'IW%':>6}")
print("-" * 50)
for label, runs in grouped.items():
    mean_bg = np.mean([r["mean_bg"] for r in runs])
    metals = np.mean([r["metal_rate"] for r in runs]) * 100
    iw = np.mean([r["iw_rate"] for r in runs]) * 100
    print(f"{label:>25} {mean_bg:>7.2f} {metals:>6.1f}% {iw:>5.1f}%")

best = max(grouped.items(), key=lambda x: np.mean([r["iw_rate"] for r in x[1]]))
best_iw = np.mean([r["iw_rate"] for r in best[1]]) * 100
bl_iw = np.mean([r["iw_rate"] for r in grouped["baseline"]]) * 100
print(f"\nBest: {best[0]} at {best_iw:.1f}% in-window (baseline: {bl_iw:.1f}%)")
