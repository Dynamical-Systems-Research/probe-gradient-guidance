"""
Train a BandGapProbe on Crystalite hidden states for self-correcting search.

Extracts atom-mean embeddings from the converged model at multiple noise levels,
trains a 2-layer MLP probe with timestep conditioning, and saves a checkpoint
compatible with BandGapProbe.from_checkpoint().

Usage:
  python scripts/train_probe.py \
    --model_checkpoint outputs/dng_alex_mp20/checkpoints/final.pt \
    --data_root data/mp20 \
    --dataset_name mp20 \
    --output_path results/self_correction/probe.pt
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, mean_absolute_error
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.crystalite.crystalite import CrystaliteModel
from src.data.mp20_tokens import MP20Tokens
from src.models.type_encoding import build_type_encoding
from scripts.self_correction import BandGapProbe, extract_atom_mean


def load_model(ckpt_path: str, device: str = "cuda") -> CrystaliteModel:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    args = ckpt.get("args", ckpt.get("config", {}))
    if hasattr(args, "__dict__"):
        args = vars(args)

    type_encoding = args.get("type_encoding", "subatomic_tokenizer_pca_24")
    enc = build_type_encoding(type_encoding, args.get("vz", 89))
    type_dim = enc.type_dim

    model = CrystaliteModel(
        d_model=args.get("d_model", 512),
        n_heads=args.get("n_heads", 16),
        n_layers=args.get("n_layers", 14),
        vz=args.get("vz", 89),
        type_dim=type_dim,
        use_distance_bias=args.get("use_distance_bias", True),
        use_edge_bias=args.get("use_edge_bias", True),
        coord_embed_mode=args.get("coord_embed_mode", "fourier"),
        lattice_embed_mode=args.get("lattice_embed_mode", "rff"),
        lattice_repr=args.get("lattice_repr", "y1"),
    )

    if "ema_state_dict" in ckpt:
        model.load_state_dict(ckpt["ema_state_dict"], strict=False)
    elif "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"], strict=False)

    return model.to(device).eval()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_checkpoint", type=str, required=True)
    parser.add_argument("--data_root", type=str, default="data/mp20")
    parser.add_argument("--dataset_name", type=str, default="mp20")
    parser.add_argument("--output_path", type=str, default="results/self_correction/probe.pt")
    parser.add_argument("--n_samples", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--probe_layer", type=int, default=-1)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--n_epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--window_low", type=float, default=4.0)
    parser.add_argument("--window_high", type=float, default=6.0)
    args = parser.parse_args()

    os.makedirs(Path(args.output_path).parent, exist_ok=True)
    device = "cuda"

    print(f"Loading model: {args.model_checkpoint}")
    model = load_model(args.model_checkpoint, device)
    d_model = model.trunk.blocks[0].attn.embed_dim
    print(f"Model: d_model={d_model}, layers={len(model.trunk.blocks)}")

    print(f"Loading dataset: {args.data_root}")
    dataset = MP20Tokens(root=args.data_root, augment_translate=False, split="val", nmax=20)

    bg_col = "dft_band_gap" if args.dataset_name == "alex_mp20" else "band_gap"
    df = pd.read_csv(f"{args.data_root}/raw/val.csv")
    bandgaps_raw = df[bg_col].values
    valid_mask = ~np.isnan(bandgaps_raw)
    valid_indices = np.where(valid_mask)[0]
    bandgaps = bandgaps_raw[valid_mask]
    print(f"Valid samples: {len(valid_indices)}")

    type_enc = build_type_encoding("subatomic_tokenizer_pca_24", 89)

    # Sample at multiple noise levels for training diversity
    sigma_values = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
    n_use = min(args.n_samples, len(valid_indices))

    all_features = []
    all_sigmas = []
    all_labels = []

    print(f"Extracting features for {n_use} samples at {len(sigma_values)} noise levels...")
    for sigma_val in sigma_values:
        for start in tqdm(range(0, n_use, args.batch_size),
                          desc=f"sigma={sigma_val:.2f}"):
            end = min(start + args.batch_size, n_use)
            batch_indices = valid_indices[start:end]
            batch_items = [dataset[i] for i in batch_indices]
            batch_bg = bandgaps[start:end]

            a0 = torch.stack([item["A0"] for item in batch_items]).to(device)
            f1 = torch.stack([item["F1"] for item in batch_items]).to(device)
            y1 = torch.stack([item["Y1"] for item in batch_items]).to(device)
            pad = torch.stack([item["pad_mask"] for item in batch_items]).to(device)

            a0_safe = a0.clamp(min=1, max=89)
            type_feats = type_enc.encode_from_A0(a0_safe, pad).to(device)

            sigma = torch.full((a0.shape[0],), sigma_val, device=device)
            noise_type = torch.randn_like(type_feats) * sigma_val
            noise_frac = torch.randn_like(f1) * sigma_val
            noise_lat = torch.randn(y1.shape, device=device) * sigma_val

            type_noisy = type_feats + noise_type
            frac_noisy = f1 + noise_frac
            lat_noisy = y1 + noise_lat
            type_noisy = type_noisy * (~pad).unsqueeze(-1).float()
            frac_noisy = frac_noisy * (~pad).unsqueeze(-1).float()

            with torch.autocast("cuda", dtype=torch.bfloat16):
                atom_mean = extract_atom_mean(
                    model, type_noisy, frac_noisy, lat_noisy,
                    pad, sigma, target_layer=args.probe_layer,
                )

            all_features.append(atom_mean.cpu())
            all_sigmas.append(sigma.cpu())
            all_labels.extend(batch_bg.tolist())

    X = torch.cat(all_features, dim=0)  # (N*S, d_model)
    T = torch.cat(all_sigmas, dim=0)    # (N*S,)
    y_reg = torch.tensor(all_labels, dtype=torch.float32)
    y_cls = ((y_reg >= args.window_low) & (y_reg <= args.window_high)).float()

    print(f"Training probe: {X.shape[0]} samples, d_model={d_model}")
    print(f"  In-window: {y_cls.sum().item():.0f} ({y_cls.mean().item()*100:.1f}%)")

    n = X.shape[0]
    n_val = int(n * 0.2)
    perm = torch.randperm(n)
    train_idx, val_idx = perm[n_val:], perm[:n_val]

    probe = BandGapProbe(input_dim=d_model, hidden_dim=args.hidden_dim, include_timestep=True).cuda()
    opt = torch.optim.Adam(probe.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.n_epochs)

    train_ds = TensorDataset(
        X[train_idx].cuda(), T[train_idx].cuda(),
        y_reg[train_idx].cuda(), y_cls[train_idx].cuda()
    )
    train_loader = DataLoader(train_ds, batch_size=512, shuffle=True)

    best_val_auroc = 0.0
    best_state = None

    for epoch in range(args.n_epochs):
        probe.train()
        for xb, tb, yb_reg, yb_cls in train_loader:
            out = probe(xb, tb)
            loss_reg = nn.functional.mse_loss(out["band_gap"], yb_reg)
            loss_cls = nn.functional.binary_cross_entropy_with_logits(out["window_logit"], yb_cls)
            loss = loss_reg + loss_cls
            opt.zero_grad()
            loss.backward()
            opt.step()
        scheduler.step()

        # Validate every 20 epochs
        if (epoch + 1) % 20 == 0:
            probe.eval()
            with torch.no_grad():
                X_val = X[val_idx].cuda()
                T_val = T[val_idx].cuda()
                val_out = probe(X_val, T_val)
                val_pred_reg = val_out["band_gap"].cpu().numpy()
                val_pred_cls = torch.sigmoid(val_out["window_logit"]).cpu().numpy()

            y_val_reg = y_reg[val_idx].numpy()
            y_val_cls = y_cls[val_idx].numpy()
            mae = mean_absolute_error(y_val_reg, val_pred_reg)

            if y_val_cls.sum() > 0 and y_val_cls.sum() < len(y_val_cls):
                auroc = roc_auc_score(y_val_cls, val_pred_cls)
            else:
                auroc = 0.0

            y_insulator = (y_reg[val_idx].numpy() > 0.5).astype(np.float32)
            if y_insulator.sum() > 0 and y_insulator.sum() < len(y_insulator):
                auroc_metal = roc_auc_score(y_insulator, val_pred_reg)
            else:
                auroc_metal = 0.0

            print(f"  epoch {epoch+1:3d}: MAE={mae:.3f} window_AUROC={auroc:.3f} "
                  f"metal_AUROC={auroc_metal:.3f}")

            if auroc > best_val_auroc:
                best_val_auroc = auroc
                best_state = {k: v.cpu().clone() for k, v in probe.state_dict().items()}

    if best_state is None:
        best_state = {k: v.cpu().clone() for k, v in probe.state_dict().items()}

    ckpt = {
        "state_dict": best_state,
        "input_dim": d_model,
        "hidden_dim": args.hidden_dim,
        "include_timestep": True,
        "probe_type": "multitask_mlp",
        "probe_layer": args.probe_layer,
        "window_low": args.window_low,
        "window_high": args.window_high,
        "best_val_auroc": best_val_auroc,
    }
    torch.save(ckpt, args.output_path)
    print(f"\nProbe saved to {args.output_path}")
    print(f"Best val window AUROC: {best_val_auroc:.3f}")


if __name__ == "__main__":
    main()
