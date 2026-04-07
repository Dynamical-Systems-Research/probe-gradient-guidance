"""
Train formation energy + E_hull probes on 100K model, then run CHGNet
evaluation on 256 structures from the w=5 Pareto batch.

Outputs:
  - results/self_correction/probe_100k_fe.pt (formation energy probe)
  - results/self_correction/probe_100k_ehull.pt (E_hull probe)
  - results/chgnet_eval_w5.json (CHGNet stability evaluation)
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, mean_absolute_error
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent  # repo root (probe-gradient-guidance/)
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.metropolis import BandGapProbe, extract_atom_mean
from scripts.train_probe import load_model
from src.data.mp20_tokens import MP20Tokens
from src.models.type_encoding import build_type_encoding
from scripts.generate import guided_edm_sampler


device = "cuda"
CKPT = "outputs/dng_alex_mp20_100k/checkpoints/final.pt"


def train_probe_for_property(
    model, prop_col: str, window_lo: float, window_hi: float,
    output_path: str, prop_label: str,
):
    """Train a probe for an arbitrary property column from MP-20 val."""
    print(f"\n{'='*60}")
    print(f"Training {prop_label} probe")
    print(f"  Column: {prop_col}, Window: [{window_lo}, {window_hi}]")
    print(f"{'='*60}")

    d_model = model.trunk.blocks[0].attn.embed_dim
    dataset = MP20Tokens(root="data/mp20", augment_translate=False, split="val", nmax=20)
    df = pd.read_csv("data/mp20/raw/val.csv")
    values = df[prop_col].values
    valid_mask = ~np.isnan(values)
    valid_indices = np.where(valid_mask)[0]
    raw_values = values[valid_mask]
    print(f"  Valid samples: {len(valid_indices)}")
    print(f"  In window: {((raw_values >= window_lo) & (raw_values <= window_hi)).sum()}")

    type_enc = build_type_encoding("subatomic_tokenizer_pca_24", 89)
    sigma_values = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
    n_use = min(5000, len(valid_indices))

    all_features, all_sigmas, all_labels = [], [], []
    for sigma_val in sigma_values:
        for start in range(0, n_use, 64):
            end = min(start + 64, n_use)
            batch_indices = valid_indices[start:end]
            batch_vals = raw_values[start:end]
            items = [dataset[i] for i in batch_indices]

            a0 = torch.stack([item["A0"] for item in items]).to(device)
            f1 = torch.stack([item["F1"] for item in items]).to(device)
            y1 = torch.stack([item["Y1"] for item in items]).to(device)
            pad = torch.stack([item["pad_mask"] for item in items]).to(device)

            a0_safe = a0.clamp(min=1, max=89)
            type_feats = type_enc.encode_from_A0(a0_safe, pad).to(device)
            sigma = torch.full((a0.shape[0],), sigma_val, device=device)

            type_noisy = type_feats + torch.randn_like(type_feats) * sigma_val
            frac_noisy = f1 + torch.randn_like(f1) * sigma_val
            lat_noisy = y1 + torch.randn(y1.shape, device=device) * sigma_val
            type_noisy = type_noisy * (~pad).unsqueeze(-1).float()
            frac_noisy = frac_noisy * (~pad).unsqueeze(-1).float()

            with torch.autocast("cuda", dtype=torch.bfloat16):
                atom_mean = extract_atom_mean(
                    model, type_noisy, frac_noisy, lat_noisy,
                    pad, sigma, target_layer=-1,
                )
            all_features.append(atom_mean.cpu())
            all_sigmas.append(sigma.cpu())
            all_labels.extend(batch_vals.tolist())

    X = torch.cat(all_features, dim=0)
    T = torch.cat(all_sigmas, dim=0)
    y_reg = torch.tensor(all_labels, dtype=torch.float32)
    y_cls = ((y_reg >= window_lo) & (y_reg <= window_hi)).float()
    print(f"  Training samples: {X.shape[0]}, in-window: {y_cls.sum().item():.0f} ({y_cls.mean()*100:.1f}%)")

    n = X.shape[0]
    n_val = int(n * 0.2)
    perm = torch.randperm(n)
    train_idx, val_idx = perm[n_val:], perm[:n_val]

    probe = BandGapProbe(input_dim=d_model, hidden_dim=256, include_timestep=True).cuda()
    opt = torch.optim.Adam(probe.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=200)

    train_ds = TensorDataset(
        X[train_idx].cuda(), T[train_idx].cuda(),
        y_reg[train_idx].cuda(), y_cls[train_idx].cuda()
    )
    loader = DataLoader(train_ds, batch_size=512, shuffle=True)

    best_auroc = 0.0
    best_state = None

    for epoch in range(200):
        probe.train()
        for xb, tb, yb_reg, yb_cls in loader:
            out = probe(xb, tb)
            loss = nn.functional.mse_loss(out["band_gap"], yb_reg) + \
                   nn.functional.binary_cross_entropy_with_logits(out["window_logit"], yb_cls)
            opt.zero_grad()
            loss.backward()
            opt.step()
        scheduler.step()

        if (epoch + 1) % 40 == 0:
            probe.eval()
            with torch.no_grad():
                val_out = probe(X[val_idx].cuda(), T[val_idx].cuda())
                pred_cls = torch.sigmoid(val_out["window_logit"]).cpu().numpy()
                pred_reg = val_out["band_gap"].cpu().numpy()
            y_val_cls = y_cls[val_idx].numpy()
            y_val_reg = y_reg[val_idx].numpy()
            mae = mean_absolute_error(y_val_reg, pred_reg)
            auroc = roc_auc_score(y_val_cls, pred_cls) if y_val_cls.sum() > 0 and y_val_cls.sum() < len(y_val_cls) else 0
            print(f"  epoch {epoch+1}: MAE={mae:.3f} AUROC={auroc:.3f}")
            if auroc > best_auroc:
                best_auroc = auroc
                best_state = {k: v.cpu().clone() for k, v in probe.state_dict().items()}

    if best_state is None:
        best_state = {k: v.cpu().clone() for k, v in probe.state_dict().items()}

    ckpt = {
        "state_dict": best_state, "input_dim": d_model, "hidden_dim": 256,
        "include_timestep": True, "best_val_auroc": best_auroc,
        "property": prop_label, "window_low": window_lo, "window_high": window_hi,
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, output_path)
    print(f"  Saved to {output_path} (AUROC={best_auroc:.3f})")
    return best_auroc


def chgnet_eval_w5(model, probe_bg):
    """Generate 256 structures at w=5 and evaluate with CHGNet."""
    print(f"\n{'='*60}")
    print("CHGNet evaluation on 256 structures at w=5")
    print(f"{'='*60}")

    type_enc = build_type_encoding("subatomic_tokenizer_pca_24", 89)
    N = 256
    nmax = 20

    torch.manual_seed(42)
    n_atoms = torch.randint(4, 13, (N,))
    pad_mask = (torch.arange(nmax).unsqueeze(0).expand(N, -1)
                >= n_atoms.unsqueeze(1)).to(device)

    print("  Generating 256 structures with w=5...")
    t0 = time.time()
    result = guided_edm_sampler(
        model=model, probe=probe_bg, pad_mask=pad_mask,
        guidance_weight=5.0, guidance_start_sigma=5.0,
        type_dim=type_enc.type_dim, num_steps=100,
        sigma_min=0.002, sigma_max=80.0, rho=7.0,
        S_churn=20.0, S_min=0.0, S_max=999.0, S_noise=1.0,
        sigma_data_type=1.0, sigma_data_coord=0.25, sigma_data_lat=1.0,
        autocast_dtype=torch.bfloat16, lattice_repr="y1",
    )
    gen_time = time.time() - t0
    print(f"  Generated in {gen_time:.1f}s")

    vz = 89
    dummy_a0 = torch.arange(1, vz + 1).unsqueeze(0)
    dummy_pad = torch.zeros(1, vz, dtype=torch.bool)
    codebook = type_enc.encode_from_A0(dummy_a0, dummy_pad).squeeze(0).to(device)
    logits_norm = torch.nn.functional.normalize(result["type"].float(), dim=-1)
    codebook_norm = torch.nn.functional.normalize(codebook.float(), dim=-1)
    sim = torch.einsum("bnd,ed->bne", logits_norm, codebook_norm)
    decoded = sim.argmax(dim=-1) + 1
    decoded = torch.where(~pad_mask, decoded, torch.zeros_like(decoded))

    from pymatgen.core import Structure, Lattice
    from pymatgen.core.periodic_table import Element

    real_mask = ~pad_mask
    decoded_np = decoded.cpu().numpy()
    frac_np = ((result["frac"] + 0.5) % 1.0).cpu().float().numpy()
    lat_np = result["lat"].cpu().float().numpy()
    mask_np = real_mask.cpu().numpy()

    structures = []
    valid_indices = []
    for i in range(N):
        atoms = decoded_np[i][mask_np[i]]
        if len(atoms) == 0:
            continue
        fracs = frac_np[i][mask_np[i]]
        y1 = lat_np[i]
        try:
            a = math.exp(float(np.clip(y1[0], -5, 5)))
            b = math.exp(float(np.clip(y1[1], -5, 5)))
            c = math.exp(float(np.clip(y1[2], -5, 5)))
            alpha = math.degrees(math.acos(float(np.clip(y1[3], -0.999, 0.999))))
            beta = math.degrees(math.acos(float(np.clip(y1[4], -0.999, 0.999))))
            gamma = math.degrees(math.acos(float(np.clip(y1[5], -0.999, 0.999))))
            lattice = Lattice.from_parameters(a, b, c, alpha, beta, gamma)
            vol_per_atom = lattice.volume / len(atoms)
            if not (5 <= vol_per_atom <= 500):
                continue
            species = [Element.from_Z(int(z)) for z in atoms]
            struct = Structure(lattice, species, fracs.tolist())
            if len(struct) > 1:
                dists = struct.distance_matrix
                np.fill_diagonal(dists, 999.0)
                if dists.min() <= 0.5:
                    continue
            structures.append(struct)
            valid_indices.append(i)
        except Exception:
            continue

    print(f"  Valid structures for CHGNet: {len(structures)} / {N}")

    print("  Loading CHGNet...")
    from chgnet.model.model import CHGNet
    chgnet = CHGNet.load()

    print(f"  Evaluating {len(structures)} structures with CHGNet (CPU, ~5 min/32)...")
    results = []
    t0 = time.time()
    for i, struct in enumerate(structures):
        try:
            prediction = chgnet.predict_structure(struct)
            e_per_atom = float(prediction["e"]) / len(struct)
            results.append({
                "idx": valid_indices[i],
                "formula": struct.composition.reduced_formula,
                "n_atoms": len(struct),
                "e_per_atom_chgnet": e_per_atom,
                "volume_per_atom": struct.volume / len(struct),
            })
        except Exception as e:
            results.append({
                "idx": valid_indices[i],
                "formula": struct.composition.reduced_formula if hasattr(struct, 'composition') else "?",
                "error": str(e),
            })
        if (i + 1) % 32 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(structures) - i - 1) / rate
            print(f"    {i+1}/{len(structures)} ({rate:.1f}/s, ETA {eta:.0f}s)")

    total_time = time.time() - t0
    print(f"  CHGNet evaluation done in {total_time:.0f}s")

    valid_results = [r for r in results if "e_per_atom_chgnet" in r]
    if valid_results:
        energies = [r["e_per_atom_chgnet"] for r in valid_results]
        print(f"\n  CHGNet Results ({len(valid_results)} evaluated):")
        print(f"    Mean energy/atom: {np.mean(energies):.3f} eV")
        print(f"    Std energy/atom: {np.std(energies):.3f} eV")
        print(f"    Min energy/atom: {np.min(energies):.3f} eV")
        print(f"    Max energy/atom: {np.max(energies):.3f} eV")
        # Rough stability proxy: lower energy/atom = more stable
        # Typical stable materials: -3 to -8 eV/atom
        # Highly unstable: > -1 eV/atom
        reasonable = [r for r in valid_results if r["e_per_atom_chgnet"] < -1.0]
        print(f"    Reasonable energy (< -1 eV/atom): {len(reasonable)} ({len(reasonable)/len(valid_results)*100:.1f}%)")
        very_stable = [r for r in valid_results if r["e_per_atom_chgnet"] < -3.0]
        print(f"    Very stable (< -3 eV/atom): {len(very_stable)} ({len(very_stable)/len(valid_results)*100:.1f}%)")

    out = {
        "config": {"weight": 5.0, "N": N, "seed": 42, "model": CKPT},
        "n_valid_structures": len(structures),
        "n_evaluated": len(valid_results),
        "n_errors": len(results) - len(valid_results),
        "results": results,
        "generation_time_s": gen_time,
        "chgnet_time_s": total_time,
    }
    out_path = "results/chgnet_eval_w5.json"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Saved to {out_path}")
    return out


def main():
    print("Loading 100K model...")
    model = load_model(CKPT, device)

    # 1. Formation energy probe
    fe_auroc = train_probe_for_property(
        model, prop_col="formation_energy_per_atom",
        window_lo=-999.0, window_hi=-1.5,
        output_path="results/self_correction/probe_100k_fe.pt",
        prop_label="formation_energy",
    )

    # 2. E_hull probe
    ehull_auroc = train_probe_for_property(
        model, prop_col="e_above_hull",
        window_lo=-999.0, window_hi=0.1,
        output_path="results/self_correction/probe_100k_ehull.pt",
        prop_label="e_above_hull",
    )

    # 3. Load band-gap probe for guided generation (trained on 100K model by pipeline_100k.sh Stage 2)
    bg_probe_path = "results/self_correction/probe_100k.pt"
    if not Path(bg_probe_path).exists():
        raise FileNotFoundError(
            f"{bg_probe_path} not found. Run pipeline_100k.sh first to train the band-gap probe on the 100K model."
        )
    probe_bg = BandGapProbe.from_checkpoint(bg_probe_path).to(device)

    # 4. CHGNet evaluation
    chgnet_results = chgnet_eval_w5(model, probe_bg)

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Formation energy probe AUROC: {fe_auroc:.3f}")
    print(f"E_hull probe AUROC: {ehull_auroc:.3f}")
    print(f"Band-gap probe AUROC: (from pipeline) see probe_100k.pt")
    print(f"CHGNet: {chgnet_results['n_valid_structures']} valid structures evaluated")
    print(f"Done.")


if __name__ == "__main__":
    main()
