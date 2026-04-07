"""Crystalite persistent generation server.

Loads the balanced 100K model once, delegates to guided_sampler,
masked_decode, decode_structures, and score_bandgap.

Deploy:
    cd <crystalite_root>
    python scripts/serve.py --port 8100
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

CRYSTALITE_ROOT = Path.home() / "crystalite"
sys.path.insert(0, str(CRYSTALITE_ROOT))
sys.path.insert(0, str(CRYSTALITE_ROOT / "src"))
sys.path.insert(0, str(CRYSTALITE_ROOT / "scripts"))

logger = logging.getLogger("crystalite_server")


class GenerateRequest(BaseModel):
    element_constraints: list[str] = Field(default_factory=list)
    element_exclusions: list[str] = Field(default_factory=list)
    element_boosts: dict[str, float] = Field(default_factory=dict)
    property_targets: dict[str, list[float]] = Field(default_factory=dict)
    guidance_weight: float = 3.0
    n_candidates: int = 10
    seed: int = 42


class BatchGenerateRequest(BaseModel):
    requests: list[GenerateRequest]


class CandidateResult(BaseModel):
    atomic_numbers: list[int]
    fractional_coordinates: list[list[float]]
    lattice: dict[str, float]
    formula: str
    probe_scores: dict[str, float]


class GenerateResponse(BaseModel):
    candidates: list[CandidateResult]
    generation_time_ms: float


class BatchGenerateResponse(BaseModel):
    results: list[GenerateResponse]


class ChgnetStructure(BaseModel):
    atomic_numbers: list[int]
    fractional_coordinates: list[list[float]]
    lattice: dict[str, float]


class ChgnetRequest(BaseModel):
    structures: list[ChgnetStructure]
    relax: bool = True
    fmax: float = 0.1
    steps: int = 250
    relax_cell: bool = True


class ChgnetResult(BaseModel):
    energy_per_atom: float | None = None
    forces_norm: float | None = None
    max_force: float | None = None
    stress_norm: float | None = None
    volume_strain: float | None = None
    converged: bool | None = None
    relaxed_atomic_numbers: list[int] | None = None
    relaxed_fractional_coordinates: list[list[float]] | None = None
    relaxed_lattice: dict[str, float] | None = None
    error: str | None = None


class ChgnetResponse(BaseModel):
    results: list[ChgnetResult]



app = FastAPI(title="Crystalite Generation Server")
_state: dict[str, Any] = {}

NOBLE_GAS_Z = {2, 10, 18, 36, 54}  # He, Ne, Ar, Kr, Xe
NEIGHBOR_DISTANCE_CUTOFF = 3.5
MIN_INTERATOMIC_DISTANCE = 0.5
OVERGENERATE_FACTOR = 4
MAX_GENERATION_BATCH = 64


def _build_pad_mask(n_atoms: torch.Tensor, nmax: int) -> torch.Tensor:
    """Return a padding mask with True on padded positions."""
    positions = torch.arange(nmax, device=n_atoms.device).unsqueeze(0)
    return positions >= n_atoms.unsqueeze(1)


def _load_models(model_ckpt: str, fe_probe_ckpt: str, bg_probe_ckpt: str | None, device: str) -> None:
    from models.type_encoding import build_type_encoding
    from self_correction import BandGapProbe
    from train_probe import load_model

    logger.info("Loading model: %s", model_ckpt)
    model = load_model(model_ckpt, device)
    model.eval()

    logger.info("Loading FE probe: %s", fe_probe_ckpt)
    fe_probe = BandGapProbe.from_checkpoint(fe_probe_ckpt).to(device)
    fe_probe.eval()

    bg_probe = None
    if bg_probe_ckpt:
        logger.info("Loading BG probe: %s", bg_probe_ckpt)
        bg_probe = BandGapProbe.from_checkpoint(bg_probe_ckpt).to(device)
        bg_probe.eval()

    type_enc = build_type_encoding("subatomic_tokenizer_pca_24", 89)

    _state.update(model=model, fe_probe=fe_probe, bg_probe=bg_probe,
                  type_enc=type_enc, device=device)
    logger.info("Ready.")


def _get_chgnet_components() -> tuple[Any, Any]:
    chgnet = _state.get("chgnet_model")
    relaxer = _state.get("chgnet_relaxer")
    if chgnet is None or relaxer is None:
        from chgnet.model import CHGNet
        from chgnet.model.dynamics import StructOptimizer

        chgnet = CHGNet.load()
        relaxer = StructOptimizer(model=chgnet, use_device=None)
        _state["chgnet_model"] = chgnet
        _state["chgnet_relaxer"] = relaxer
    return chgnet, relaxer


def _lattice_to_dict(structure: Any) -> dict[str, float]:
    lattice = structure.lattice
    return {
        "a": round(float(lattice.a), 4),
        "b": round(float(lattice.b), 4),
        "c": round(float(lattice.c), 4),
        "alpha": round(float(lattice.alpha), 2),
        "beta": round(float(lattice.beta), 2),
        "gamma": round(float(lattice.gamma), 2),
    }


def _build_structure(
    *,
    atomic_numbers: list[int],
    fractional_coordinates: list[list[float]],
    lattice: dict[str, float],
) -> Any:
    from pymatgen.core import Element, Lattice, Structure

    return Structure(
        Lattice.from_parameters(
            lattice["a"],
            lattice["b"],
            lattice["c"],
            lattice["alpha"],
            lattice["beta"],
            lattice["gamma"],
        ),
        [Element.from_Z(z) for z in atomic_numbers],
        fractional_coordinates,
    )


def _has_all_constrained_elements(atoms: list[int], element_constraints: list[str]) -> bool:
    if not element_constraints:
        return True

    from pymatgen.core import Element

    required = {Element(symbol).Z for symbol in element_constraints}
    present = set(atoms)
    return required <= present


def _passes_structural_validity_filter(
    *,
    atomic_numbers: list[int],
    fractional_coordinates: list[list[float]],
    lattice: dict[str, float],
    element_constraints: list[str],
) -> bool:
    if not atomic_numbers:
        return False
    if not _has_all_constrained_elements(atomic_numbers, element_constraints):
        return False

    structure = _build_structure(
        atomic_numbers=atomic_numbers,
        fractional_coordinates=fractional_coordinates,
        lattice=lattice,
    )
    if structure.volume <= 0.1 or len(structure) == 0:
        return False

    try:
        distances = structure.distance_matrix
    except Exception:
        return False

    min_distance = math.inf
    for row_index in range(len(distances)):
        has_neighbor = False
        for col_index in range(len(distances)):
            if row_index == col_index:
                continue
            distance = float(distances[row_index][col_index])
            min_distance = min(min_distance, distance)
            if distance <= NEIGHBOR_DISTANCE_CUTOFF:
                has_neighbor = True
        if not has_neighbor:
            return False

    return min_distance == math.inf or min_distance >= MIN_INTERATOMIC_DISTANCE



def _generate_candidates(req: GenerateRequest) -> GenerateResponse:
    import time as _time

    from guided_sampler import guided_edm_sampler
    from hybrid_constrained import masked_decode, score_bandgap
    from pymatgen.core import Element
    from run_sweep import decode_structures

    model = _state["model"]
    fe_probe = _state["fe_probe"]
    bg_probe = _state["bg_probe"]
    type_enc = _state["type_enc"]
    device = _state["device"]
    n = req.n_candidates
    # The current Crystalite probe-scoring path is only reliable up to batch size 20.
    # Larger overgenerated batches trigger an upstream shape bug and 500 the request.
    generation_batch_size = min(20, MAX_GENERATION_BATCH, max(n * OVERGENERATE_FACTOR, n))

    t0 = _time.monotonic()

    # Crystalite expects True on padded positions, not real atoms.
    rng = torch.Generator(device=device).manual_seed(req.seed)
    n_atoms = torch.randint(4, 13, (generation_batch_size,), generator=rng, device=device)
    pad_mask = _build_pad_mask(n_atoms, 20)

    exclude_z: set[int] = set(NOBLE_GAS_Z)
    for elem in req.element_exclusions:
        try:
            exclude_z.add(Element(elem).Z)
        except (ValueError, KeyError):
            pass
    if req.element_constraints:
        allowed_z = set()
        for elem in req.element_constraints:
            try:
                allowed_z.add(Element(elem).Z)
            except (ValueError, KeyError):
                pass
        exclude_z |= (set(range(1, 90)) - allowed_z)

    with torch.no_grad():
        result = guided_edm_sampler(
            model=model, probe=fe_probe, pad_mask=pad_mask,
            guidance_weight=req.guidance_weight, guidance_start_sigma=5.0,
            type_dim=type_enc.type_dim, num_steps=100,
            sigma_min=0.002, sigma_max=80.0, rho=7.0,
            S_churn=20.0, S_min=0.0, S_max=999.0, S_noise=1.0,
            sigma_data_type=1.0, sigma_data_coord=0.25, sigma_data_lat=1.0,
            autocast_dtype=torch.bfloat16, lattice_repr="y1",
        )

    boost_z: set[int] = set()
    boost_strength = 5.0
    if req.element_boosts:
        for elem, strength in req.element_boosts.items():
            try:
                boost_z.add(Element(elem).Z)
            except (ValueError, KeyError):
                pass
        boost_strength = max(req.element_boosts.values()) if req.element_boosts else 5.0
    elif req.element_constraints and len(req.element_constraints) > 1:
        # Auto-boost non-anion constrained elements
        # Anions (O, S, Se, N, F, Cl) dominate the model's prior in guided mode
        anion_z = {Element(e).Z for e in ["O", "S", "Se", "N", "F", "Cl"] if e in req.element_constraints}
        for elem in req.element_constraints:
            try:
                z = Element(elem).Z
                if z not in anion_z:
                    boost_z.add(z)
            except (ValueError, KeyError):
                pass
        boost_strength = 5.0

    decoded = masked_decode(
        result["type"], pad_mask, type_enc,
        exclude_z=exclude_z if exclude_z else None,
        boost_z=boost_z if boost_z else None,
        boost_strength=boost_strength,
    )

    structures = decode_structures(result, pad_mask, type_enc, lattice_repr="y1")

    # Override atoms with constrained decode (decode_structures uses unconstrained)
    real_mask = ~pad_mask
    for i, s in enumerate(structures):
        s["atoms"] = decoded[i][real_mask[i]].cpu().tolist()

    fe_scores = score_bandgap(model, fe_probe, result, pad_mask, device)
    bg_scores = score_bandgap(model, bg_probe, result, pad_mask, device) if bg_probe else None

    elapsed_ms = (_time.monotonic() - t0) * 1000

    candidates = []
    filtered_missing_elements = 0
    filtered_invalid_geometry = 0
    for i, s in enumerate(structures):
        atoms = [int(z) for z in s["atoms"] if z > 0]
        if not atoms:
            continue

        frac = s["frac_coords"][:len(atoms)]
        L = np.array(s["lattice_matrix"])
        a = float(np.linalg.norm(L[0]))
        b = float(np.linalg.norm(L[1]))
        c = float(np.linalg.norm(L[2]))
        alpha = float(np.degrees(np.arccos(np.clip(np.dot(L[1], L[2]) / (b * c), -1, 1))))
        beta = float(np.degrees(np.arccos(np.clip(np.dot(L[0], L[2]) / (a * c), -1, 1))))
        gamma = float(np.degrees(np.arccos(np.clip(np.dot(L[0], L[1]) / (a * b), -1, 1))))

        counts = Counter(atoms)
        gcd = math.gcd(*counts.values()) if counts else 1
        formula = "".join(
            f"{Element.from_Z(z).symbol}{c_ // gcd if c_ // gcd > 1 else ''}"
            for z, c_ in sorted(counts.items())
        )

        scores = {"formation_energy": float(fe_scores[i])}
        if bg_scores is not None:
            scores["band_gap"] = float(bg_scores[i])

        lattice = {"a": round(a, 4), "b": round(b, 4), "c": round(c, 4),
                   "alpha": round(alpha, 2), "beta": round(beta, 2), "gamma": round(gamma, 2)}
        fractional_coordinates = [[round(x, 6) for x in coord] for coord in frac]

        if not _has_all_constrained_elements(atoms, req.element_constraints):
            filtered_missing_elements += 1
            continue
        if not _passes_structural_validity_filter(
            atomic_numbers=atoms,
            fractional_coordinates=fractional_coordinates,
            lattice=lattice,
            element_constraints=req.element_constraints,
        ):
            filtered_invalid_geometry += 1
            continue

        candidates.append(CandidateResult(
            atomic_numbers=atoms,
            fractional_coordinates=fractional_coordinates,
            lattice=lattice,
            formula=formula,
            probe_scores=scores,
        ))
        if len(candidates) >= n:
            break

    logger.info(
        "generate request=%s requested=%d raw=%d kept=%d filtered_missing_elements=%d filtered_invalid_geometry=%d",
        req.element_constraints,
        n,
        len(structures),
        len(candidates),
        filtered_missing_elements,
        filtered_invalid_geometry,
    )

    return GenerateResponse(candidates=candidates, generation_time_ms=round(elapsed_ms, 1))



def _evaluate_chgnet(req: ChgnetRequest) -> list[ChgnetResult]:
    from pymatgen.core import Element, Lattice, Structure

    chgnet, relaxer = _get_chgnet_components()
    results = []
    for s in req.structures:
        try:
            lat = Lattice.from_parameters(
                s.lattice["a"], s.lattice["b"], s.lattice["c"],
                s.lattice["alpha"], s.lattice["beta"], s.lattice["gamma"],
            )
            species = [Element.from_Z(z) for z in s.atomic_numbers]
            initial_structure = Structure(lat, species, s.fractional_coordinates)
            final_structure = initial_structure
            if req.relax:
                relaxed = relaxer.relax(
                    initial_structure,
                    fmax=req.fmax,
                    steps=req.steps,
                    relax_cell=req.relax_cell,
                    verbose=False,
                )
                final_structure = relaxed["final_structure"]
            pred = chgnet.predict_structure(final_structure)
            force_array = np.array(pred["f"], dtype=float)
            max_force = float(np.linalg.norm(force_array, axis=1).max()) if force_array.size else 0.0
            initial_volume = max(float(initial_structure.volume), 1e-9)
            volume_strain = abs(float(final_structure.volume) - float(initial_structure.volume)) / initial_volume
            results.append(ChgnetResult(
                energy_per_atom=round(float(pred["e"]) / len(s.atomic_numbers), 6),
                forces_norm=round(float(np.linalg.norm(force_array)), 6),
                max_force=round(max_force, 6),
                stress_norm=round(float(np.linalg.norm(pred["s"])), 6),
                volume_strain=round(volume_strain, 6),
                converged=None if not req.relax else max_force <= req.fmax,
                relaxed_atomic_numbers=[int(site.specie.Z) for site in final_structure],
                relaxed_fractional_coordinates=[
                    [round(float(x), 6) for x in site.frac_coords]
                    for site in final_structure
                ],
                relaxed_lattice=_lattice_to_dict(final_structure),
            ))
        except Exception as exc:
            results.append(ChgnetResult(error=str(exc)))
    return results



@app.get("/health")
def health():
    if "model" not in _state:
        raise HTTPException(503, "Not loaded")
    return {"status": "ok", "device": _state["device"]}


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    return _generate_candidates(req)


@app.post("/generate_batch", response_model=BatchGenerateResponse)
def generate_batch(req: BatchGenerateRequest):
    return BatchGenerateResponse(results=[_generate_candidates(r) for r in req.requests])


@app.post("/evaluate/chgnet", response_model=ChgnetResponse)
def evaluate_chgnet(req: ChgnetRequest):
    return ChgnetResponse(results=_evaluate_chgnet(req))



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8100)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument(
        "--model-checkpoint",
        default=str(CRYSTALITE_ROOT / "outputs/dng_balanced_100k/checkpoints/final.pt"),
    )
    parser.add_argument(
        "--fe-probe",
        default=str(CRYSTALITE_ROOT / "results/self_correction/probe_balanced_fe.pt"),
    )
    parser.add_argument(
        "--bg-probe",
        default=str(CRYSTALITE_ROOT / "results/self_correction/probe_balanced_bandgap.pt"),
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    _load_models(args.model_checkpoint, args.fe_probe, args.bg_probe, args.device)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
