"""Decode sampler output tensors to structure dictionaries."""
from __future__ import annotations

import numpy as np
import torch

from src.models.lattice_repr import lattice_latent_to_y1


def decode_structures(
    result: dict[str, torch.Tensor],
    pad_mask: torch.Tensor,
    type_enc,
    lattice_repr: str = "y1",
) -> list[dict]:
    """Decode sampler output to list of structure dicts with atoms, coords, lattice."""
    type_logits = result["type"]
    frac = result["frac"]
    lat = result["lat"]

    decoded_a0 = type_enc.decode_logits_to_A0(type_logits, pad_mask)
    real_mask = ~pad_mask

    structures = []
    for i in range(type_logits.shape[0]):
        n = real_mask[i].sum().item()
        atoms = decoded_a0[i][:n].cpu().tolist()
        coords = (frac[i][:n].cpu() + 0.5).fmod(1.0).numpy()
        lattice_vec = lat[i].cpu().unsqueeze(0)
        y1_val = lattice_latent_to_y1(lattice_vec, lattice_repr=lattice_repr)
        lengths = torch.exp(y1_val[0, :3]).clamp(min=0.5, max=50.0)
        cos_angles = y1_val[0, 3:].clamp(-0.99, 0.99)
        a, b, c = lengths.tolist()
        cos_a, cos_b, cos_g = cos_angles.tolist()
        sin_g = max((1 - cos_g**2)**0.5, 1e-6)
        val = max(1 - cos_a**2 - cos_b**2 - cos_g**2 + 2*cos_a*cos_b*cos_g, 1e-6)
        L = np.array([
            [a, 0, 0],
            [b * cos_g, b * sin_g, 0],
            [c * cos_b, c * (cos_a - cos_b * cos_g) / sin_g,
             c * (val**0.5) / sin_g],
        ])

        structures.append({
            "atoms": atoms,
            "frac_coords": coords.tolist(),
            "lattice_matrix": L.tolist(),
            "n_atoms": n,
        })
    return structures
