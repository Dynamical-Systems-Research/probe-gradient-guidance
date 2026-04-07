"""
Experiment 8: Pareto Frontier — Targeting vs Diversity

Generate 1024 structures at each guidance weight (w=0,1,3,5,10,15),
3 seeds, evaluate both targeting AND diversity metrics.

Tests whether probe-gradient guidance is Pareto-optimal:
improving targeting without sacrificing compositional diversity.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter, defaultdict
from functools import reduce
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pymatgen.core import Lattice, Structure
from pymatgen.core.periodic_table import Element

from src.crystalite.sampler import edm_sampler
from src.models.type_encoding import build_type_encoding
from scripts.metropolis import BandGapProbe, extract_atom_mean
from scripts.train_probe import load_model
from scripts.generate import guided_edm_sampler



def reduced_formula(atomic_numbers: list[int]) -> str:
    """Reduced formula string from atomic number list. E.g. [75,75,72,74] -> 'HfRe2W'."""
    if not atomic_numbers:
        return ""
    counts = Counter(atomic_numbers)
    g = reduce(math.gcd, counts.values())
    parts = []
    for z in sorted(counts, key=lambda z: Element.from_Z(z).symbol):
        sym = Element.from_Z(z).symbol
        c = counts[z] // g
        parts.append(f"{sym}{c}" if c > 1 else sym)
    return "".join(parts)


def chemical_system(atomic_numbers: list[int]) -> str:
    """Element-set string. E.g. [75,75,72,74] -> 'Hf-Re-W'."""
    elements = sorted({Element.from_Z(z).symbol for z in atomic_numbers})
    return "-".join(elements)


def element_entropy(all_atoms: list[list[int]]) -> float:
    """Shannon entropy (bits) over element frequencies across all structures."""
    counts: Counter = Counter()
    total = 0
    for atoms in all_atoms:
        counts.update(atoms)
        total += len(atoms)
    if total == 0:
        return 0.0
    H = 0.0
    for c in counts.values():
        p = c / total
        if p > 0:
            H -= p * math.log2(p)
    return H


def lattice_from_y1(y1: np.ndarray) -> Lattice:
    """Y1 = [log_a, log_b, log_c, cos_alpha, cos_beta, cos_gamma] -> pymatgen Lattice."""
    a = math.exp(float(np.clip(y1[0], -5, 5)))
    b = math.exp(float(np.clip(y1[1], -5, 5)))
    c = math.exp(float(np.clip(y1[2], -5, 5)))
    alpha = math.degrees(math.acos(float(np.clip(y1[3], -0.999, 0.999))))
    beta = math.degrees(math.acos(float(np.clip(y1[4], -0.999, 0.999))))
    gamma = math.degrees(math.acos(float(np.clip(y1[5], -0.999, 0.999))))
    return Lattice.from_parameters(a, b, c, alpha, beta, gamma)


def check_structural_validity(
    decoded: torch.Tensor,
    frac_coords: torch.Tensor,
    lat_y1: torch.Tensor,
    pad_mask: torch.Tensor,
    sample_n: int = 256,
) -> dict:
    """Check lattice validity and interatomic distances on a subsample."""
    B = decoded.shape[0]
    idx = list(range(min(sample_n, B)))
    n_check = len(idx)

    real_mask = ~pad_mask
    decoded_np = decoded[idx].cpu().numpy()
    frac_np = ((frac_coords[idx] + 0.5) % 1.0).cpu().float().numpy()
    lat_np = lat_y1[idx].cpu().float().numpy()
    mask_np = real_mask[idx].cpu().numpy()

    valid_lattice = 0
    valid_geometry = 0

    for i in range(n_check):
        atoms = decoded_np[i][mask_np[i]]
        if len(atoms) == 0:
            continue
        fracs = frac_np[i][mask_np[i]]
        y1 = lat_np[i]

        try:
            lattice = lattice_from_y1(y1)
            vol_per_atom = lattice.volume / len(atoms)
            if not (5 <= vol_per_atom <= 500):
                continue
            valid_lattice += 1
        except Exception:
            continue

        try:
            species = [Element.from_Z(int(z)) for z in atoms]
            struct = Structure(lattice, species, fracs.tolist())
            if len(struct) > 1:
                dists = struct.distance_matrix
                np.fill_diagonal(dists, 999.0)
                if dists.min() > 0.5:
                    valid_geometry += 1
            else:
                valid_geometry += 1
        except Exception:
            continue

    return {
        "valid_lattice_frac": valid_lattice / n_check,
        "valid_geometry_frac": valid_geometry / n_check,
        "n_checked": n_check,
    }



def decode_to_atoms(type_logits: torch.Tensor, pad_mask: torch.Tensor,
                    type_enc) -> torch.Tensor:
    """Cosine-similarity decode from continuous type embeddings to atomic numbers."""
    vz = 89
    dummy_a0 = torch.arange(1, vz + 1).unsqueeze(0)
    dummy_pad = torch.zeros(1, vz, dtype=torch.bool)
    codebook = type_enc.encode_from_A0(dummy_a0, dummy_pad).squeeze(0).to(type_logits.device)

    logits_norm = torch.nn.functional.normalize(type_logits.float(), dim=-1)
    codebook_norm = torch.nn.functional.normalize(codebook.float(), dim=-1)
    sim = torch.einsum("bnd,ed->bne", logits_norm, codebook_norm)

    decoded = sim.argmax(dim=-1) + 1
    return torch.where(~pad_mask, decoded, torch.zeros_like(decoded))


def score_bandgap(model, probe, result, pad_mask, device="cuda"):
    sigma = torch.full((result["type"].shape[0],), 0.01, device=device)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        am = extract_atom_mean(model, result["type"], result["frac"] + 0.5,
                               result["lat"], pad_mask, sigma, -1)
        out = probe(am.to(device), sigma)
    return out["band_gap"].float().cpu().numpy()



def load_training_systems(cache_path: str = "results/train_chemical_systems.txt") -> set[str]:
    p = Path(cache_path)
    if not p.exists():
        raise FileNotFoundError(
            f"{cache_path} not found. Run: python -c 'extract systems from train.csv' first."
        )
    systems = set(p.read_text().strip().split("\n"))
    print(f"Loaded {len(systems)} training chemical systems from cache")
    return systems



def main():
    parser = argparse.ArgumentParser(description="Pareto sweep: targeting vs diversity")
    parser.add_argument("--model_checkpoint", type=str,
                        default="outputs/dng_alex_mp20/checkpoints/final.pt")
    parser.add_argument("--probe_path", type=str,
                        default="results/self_correction/probe.pt")
    parser.add_argument("--output_dir", type=str, default="results/pareto_sweep")
    args = parser.parse_args()

    device = "cuda"

    print(f"Loading model: {args.model_checkpoint}")
    print(f"Loading probe: {args.probe_path}")
    model = load_model(args.model_checkpoint, device)
    probe = BandGapProbe.from_checkpoint(args.probe_path).to(device)
    type_enc = build_type_encoding("subatomic_tokenizer_pca_24", 89)
    train_systems = load_training_systems()

    N = 1024
    nmax = 20
    weights = [0, 1, 3, 5, 10, 15]
    seeds = [42, 123, 456]

    sampler_kw = dict(
        type_dim=type_enc.type_dim, num_steps=100,
        sigma_min=0.002, sigma_max=80.0, rho=7.0,
        S_churn=20.0, S_min=0.0, S_max=999.0, S_noise=1.0,
        sigma_data_type=1.0, sigma_data_coord=0.25, sigma_data_lat=1.0,
        autocast_dtype=torch.bfloat16, lattice_repr="y1",
    )

    all_results = []

    hdr = (f"{'w':>3} {'seed':>4} {'IW%':>6} {'Uniq%':>6} {'H(elem)':>7} "
           f"{'#Sys':>5} {'Nov%':>6} {'VLat%':>6} {'VGeo%':>6} "
           f"{'Metal%':>6} {'MeanBG':>7} {'t(s)':>5}")
    print(f"\n{hdr}\n{'-'*len(hdr)}")

    for w in weights:
        for seed in seeds:
            t0 = time.time()

            torch.manual_seed(seed)
            n_atoms = torch.randint(4, 13, (N,))
            pad_mask = (torch.arange(nmax).unsqueeze(0).expand(N, -1)
                        >= n_atoms.unsqueeze(1)).to(device)

            if w == 0:
                result = edm_sampler(model=model, pad_mask=pad_mask, **sampler_kw)
            else:
                result = guided_edm_sampler(
                    model=model, probe=probe, pad_mask=pad_mask,
                    guidance_weight=float(w), guidance_start_sigma=5.0,
                    **sampler_kw,
                )

            gen_time = time.time() - t0

            decoded = decode_to_atoms(result["type"], pad_mask, type_enc)
            real_mask = ~pad_mask

            bg = score_bandgap(model, probe, result, pad_mask)

            # Per-structure composition
            dec_np = decoded.cpu().numpy()
            msk_np = real_mask.cpu().numpy()
            atom_lists, formulas, systems_list = [], [], []
            for i in range(N):
                atoms = dec_np[i][msk_np[i]].tolist()
                atom_lists.append(atoms)
                formulas.append(reduced_formula(atoms))
                systems_list.append(chemical_system(atoms))

            iw_rate = float(((bg >= 4.0) & (bg <= 6.0)).mean())
            metal_rate = float((bg < 0.5).mean())
            unique_formulas = len(set(formulas))
            comp_uniq = unique_formulas / N
            ent = element_entropy(atom_lists)
            n_sys = len(set(systems_list))
            novelty = sum(1 for s in systems_list if s not in train_systems) / N

            validity = check_structural_validity(
                decoded, result["frac"], result["lat"], pad_mask, sample_n=256,
            )

            total_t = time.time() - t0
            mean_bg = float(bg.mean())

            print(f"{w:>3} {seed:>4} {iw_rate*100:>5.1f}% {comp_uniq*100:>5.1f}% "
                  f"{ent:>7.3f} {n_sys:>5} {novelty*100:>5.1f}% "
                  f"{validity['valid_lattice_frac']*100:>5.1f}% "
                  f"{validity['valid_geometry_frac']*100:>5.1f}% "
                  f"{metal_rate*100:>5.1f}% {mean_bg:>7.2f} {total_t:>4.0f}s")

            # Element distribution: top-10 elements
            elem_counts = Counter(z for atoms in atom_lists for z in atoms)
            top10 = [(Element.from_Z(z).symbol, c)
                     for z, c in elem_counts.most_common(10)]

            all_results.append({
                "weight": w, "seed": seed,
                "iw_rate": iw_rate, "comp_uniqueness": comp_uniq,
                "element_entropy": ent, "n_unique_systems": n_sys,
                "n_unique_formulas": unique_formulas,
                "novelty": novelty,
                "valid_lattice_frac": validity["valid_lattice_frac"],
                "valid_geometry_frac": validity["valid_geometry_frac"],
                "mean_bg": mean_bg, "metal_rate": metal_rate,
                "gen_time_s": gen_time, "total_time_s": total_t,
                "top10_elements": top10,
                "element_counts": {str(z): c for z, c in elem_counts.items()},
                "sample_formulas": formulas[:30],
            })

    grouped = defaultdict(list)
    for r in all_results:
        grouped[r["weight"]].append(r)

    print(f"\n{'='*75}")
    print("MEAN +/- STD ACROSS 3 SEEDS")
    print(f"{'='*75}")
    shdr = (f"{'w':>3} {'IW%':>10} {'Uniq%':>10} {'H(elem)':>10} "
            f"{'#Sys':>6} {'Nov%':>10} {'Metal%':>7}")
    print(f"{shdr}\n{'-'*len(shdr)}")

    summary = []
    for w in weights:
        runs = grouped[w]
        def ms(key):
            vals = [r[key] for r in runs]
            return np.mean(vals), np.std(vals)

        iw_m, iw_s = ms("iw_rate")
        cu_m, cu_s = ms("comp_uniqueness")
        en_m, en_s = ms("element_entropy")
        sy_m, _ = ms("n_unique_systems")
        nv_m, nv_s = ms("novelty")
        mt_m, _ = ms("metal_rate")
        bg_m, _ = ms("mean_bg")
        vl_m, _ = ms("valid_lattice_frac")
        vg_m, _ = ms("valid_geometry_frac")

        print(f"{w:>3} {iw_m*100:>4.1f}+/-{iw_s*100:>3.1f} "
              f"{cu_m*100:>4.1f}+/-{cu_s*100:>3.1f} "
              f"{en_m:>4.2f}+/-{en_s:>3.2f} "
              f"{sy_m:>5.0f} {nv_m*100:>4.1f}+/-{nv_s*100:>3.1f} "
              f"{mt_m*100:>6.1f}%")

        summary.append({
            "weight": w,
            "iw_rate": {"mean": iw_m, "std": iw_s},
            "comp_uniqueness": {"mean": cu_m, "std": cu_s},
            "element_entropy": {"mean": en_m, "std": en_s},
            "n_unique_systems": {"mean": sy_m},
            "novelty": {"mean": nv_m, "std": nv_s},
            "metal_rate": {"mean": mt_m},
            "mean_bg": {"mean": bg_m},
            "valid_lattice_frac": {"mean": vl_m},
            "valid_geometry_frac": {"mean": vg_m},
        })

    bl = summary[0]
    print(f"\n{'='*75}")
    print("PARETO ASSESSMENT (vs baseline w=0)")
    print(f"{'='*75}")
    for s in summary[1:]:
        iw_d = s["iw_rate"]["mean"] - bl["iw_rate"]["mean"]
        cu_d = s["comp_uniqueness"]["mean"] - bl["comp_uniqueness"]["mean"]
        tag = ("DOMINATES" if iw_d > 0.01 and cu_d >= -0.02
               else "TRADEOFF" if iw_d > 0.01 and cu_d < -0.02
               else "NEUTRAL")
        print(f"  w={s['weight']:>2}: IW {iw_d:>+6.1%}, Uniq {cu_d:>+6.1%}  -> {tag}")

    # Element distribution comparison
    print(f"\n{'='*75}")
    print("TOP-10 ELEMENTS BY WEIGHT (seed=42)")
    print(f"{'='*75}")
    for w in [0, 5, 10]:
        r = [x for x in all_results if x["weight"] == w and x["seed"] == 42][0]
        elems = " ".join(f"{sym}:{c}" for sym, c in r["top10_elements"])
        print(f"  w={w:>2}: {elems}")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "raw_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved to {out}/")


if __name__ == "__main__":
    main()
