#!/usr/bin/env python3
"""
MatterGen Self-Correcting Search — Full Frontier Sweep
Reproduces Goodfire's test-time self-verification across gamma values and seeds.

Phases:
  1. Cross-gamma robustness: gamma=2.0, seed=2026
  2. Multi-seed at gamma=1.0: seeds 42, 137 (seed 2026 already done)
  3. Pareto frontier: gamma=0.5, 1.5, 3.0 at seed=2026
  4. Aggregate all results + existing v2 results into frontier_summary.json

All generation runs via Docker on GPU.
Evaluation via host-side Python envs (CPU).
"""

import json
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

MATTERGEN_ROOT = Path(os.environ.get("MATTERGEN_ROOT", str(Path.home() / "mattergen")))
RESULTS_ROOT = MATTERGEN_ROOT / "results"
REFERENCE_DATASET = MATTERGEN_ROOT / "data-release/alex-mp/reference_TRI2024correction.gz"
PROBE_CHECKPOINT = RESULTS_ROOT / "self_correction/window_probe_v2.pt"
MODEL_PATH = MATTERGEN_ROOT / "checkpoints/dft_band_gap"
DOCKER_IMAGE = "mattergen-canonical:py310"

MATTERSIM_PYTHON = "/tmp/mattersim-py310-fixed/bin/python"
MATGL_PYTHON = "/tmp/matgl-eval/bin/python"
VENV_PYTHON = str(MATTERGEN_ROOT / ".venv/bin/python")
MATGL_SCORE_SCRIPT = str(MATTERGEN_ROOT / "mattergen/scripts/score_bandgap_matgl.py")

# Self-correction hyperparameters (from v2 corrected run)
SC_PARAMS = {
    "max_scoring_t": 0.05,
    "center_weight": 0.35,
    "in_window_weight": 0.5,
    "window_exit_penalty": 1.0,
}

LOG_FILE = RESULTS_ROOT / "frontier_sweep.log"


@dataclass
class ExperimentArm:
    name: str
    gamma: float
    seed: int
    self_correct: bool
    num_samples: int = 32
    batch_size: int = 4
    target_band_gap: float = 5.0

    @property
    def output_dir(self) -> Path:
        sc_tag = "sc" if self.self_correct else "base"
        return RESULTS_ROOT / f"frontier_g{self.gamma}_s{self.seed}_{sc_tag}"

    @property
    def num_batches(self) -> int:
        return self.num_samples // self.batch_size


def log(msg: str):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


def run_cmd(cmd: str, timeout: int = 1800, desc: str = "") -> subprocess.CompletedProcess:
    """Run a shell command, log it, and return the result."""
    if desc:
        log(f"  CMD ({desc}): {cmd[:200]}...")
    else:
        log(f"  CMD: {cmd[:200]}...")
    result = subprocess.run(
        cmd, shell=True, capture_output=True, text=True, timeout=timeout
    )
    if result.returncode != 0:
        log(f"  STDERR: {result.stderr[-500:]}")
    return result


def generate(arm: ExperimentArm) -> bool:
    """Run MatterGen generation via Docker on GPU."""
    out = arm.output_dir
    if (out / "generated_crystals.extxyz").exists():
        log(f"  SKIP generation — {out}/generated_crystals.extxyz already exists")
        return True

    out.mkdir(parents=True, exist_ok=True)

    docker_out = f"/workspace/mattergen/results/{out.name}"
    cmd_parts = [
        f"docker run --rm --gpus all --entrypoint python",
        f"-v {MATTERGEN_ROOT}:/workspace/mattergen",
        f"-w /workspace/mattergen",
        f"{DOCKER_IMAGE}",
        f"-m mattergen.scripts.run_smoke_sample",
        f"{docker_out}",
        f"--model_path=/workspace/mattergen/checkpoints/dft_band_gap",
        f"--target_band_gap={arm.target_band_gap}",
        f"--diffusion_guidance_factor={arm.gamma}",
        f"--batch_size={arm.batch_size}",
        f"--num_batches={arm.num_batches}",
        f"--seed={arm.seed}",
    ]
    if arm.self_correct:
        docker_probe = f"/workspace/mattergen/results/self_correction/window_probe_v2.pt"
        cmd_parts.extend([
            f"--self_correct=true",
            f"--probe_checkpoint_path={docker_probe}",
            f"--max_scoring_t={SC_PARAMS['max_scoring_t']}",
            f"--center_weight={SC_PARAMS['center_weight']}",
            f"--in_window_weight={SC_PARAMS['in_window_weight']}",
            f"--window_exit_penalty={SC_PARAMS['window_exit_penalty']}",
        ])

    cmd = " ".join(cmd_parts)
    t0 = time.time()
    result = run_cmd(cmd, timeout=3600, desc="docker generate")
    elapsed = time.time() - t0
    log(f"  Generation took {elapsed:.1f}s")

    if result.returncode != 0:
        log(f"  GENERATION FAILED: {result.stderr[-300:]}")
        return False

    # Fix Docker root-owned permissions
    run_cmd(
        f"docker run --rm -v {RESULTS_ROOT}:/workspace/results "
        f"alpine sh -lc 'chown -R 1000:1000 /workspace/results/{out.name}'",
        desc="fix permissions"
    )

    # Write timing metadata
    meta = {"generation_time_s": elapsed, "gamma": arm.gamma, "seed": arm.seed,
            "self_correct": arm.self_correct, "num_samples": arm.num_samples}
    (out / "generation_meta.json").write_text(json.dumps(meta, indent=2))

    return (out / "generated_crystals.extxyz").exists()


def score_matgl(structures_path: Path, output_path: Path, desc: str = "") -> bool:
    """Score structures with MatGL band-gap surrogate."""
    if output_path.exists():
        log(f"  SKIP MatGL — {output_path} already exists")
        return True
    cmd = (
        f"{MATGL_PYTHON} {MATGL_SCORE_SCRIPT} "
        f"--structures-path {structures_path} "
        f"--output-path {output_path}"
    )
    result = run_cmd(cmd, timeout=300, desc=f"matgl {desc}")
    return result.returncode == 0


def evaluate_mattersim(arm: ExperimentArm) -> bool:
    """Run MatterSim relaxation + TRI2024 hull evaluation."""
    out = arm.output_dir
    if (out / "relaxed.extxyz").exists() and (out / "detailed_metrics.json").exists():
        log(f"  SKIP MatterSim eval — relaxed.extxyz and detailed_metrics.json exist")
        return True

    eval_script = f"""
import sys
sys.path.insert(0, '{MATTERGEN_ROOT}')
from pathlib import Path
import ase.io
from pymatgen.io.ase import AseAtomsAdaptor
from mattergen.evaluation.evaluate import evaluate
from mattergen.evaluation.reference.reference_dataset_serializer import LMDBGZSerializer
from mattergen.evaluation.reference.correction_schemes import TRI110Compatibility2024

base = Path('{out}')
ase_atoms = ase.io.read(base / 'generated_crystals.extxyz', ':')
structures = [AseAtomsAdaptor.get_structure(x) for x in ase_atoms]
reference = LMDBGZSerializer().deserialize('{REFERENCE_DATASET}')
metrics = evaluate(
    structures=structures,
    relax=True,
    potential_load_path='MatterSim-v1.0.0-1M.pth',
    reference=reference,
    save_detailed_as=str(base / 'detailed_metrics.json'),
    structures_output_path=str(base / 'relaxed.extxyz'),
    energy_correction_scheme=TRI110Compatibility2024(),
)
import json
print(json.dumps(metrics, indent=2))
"""
    cmd = f'PYTHONPATH={MATTERGEN_ROOT} {MATTERSIM_PYTHON} -u -c "{eval_script.strip()}"'
    # Use heredoc approach for safety with quotes
    script_path = out / "_eval_mattersim.py"
    script_path.write_text(eval_script.strip())
    cmd = f"PYTHONPATH={MATTERGEN_ROOT} {MATTERSIM_PYTHON} -u {script_path}"
    result = run_cmd(cmd, timeout=600, desc="mattersim eval")
    return result.returncode == 0


def compute_energy_summary(arm: ExperimentArm) -> bool:
    """Compute stable_mask and avg_energy_above_hull from relaxed structures."""
    out = arm.output_dir
    if (out / "energy_summary.json").exists():
        log(f"  SKIP energy summary — already exists")
        return True

    script = f"""
import sys
sys.path.insert(0, '{MATTERGEN_ROOT}')
from pathlib import Path
import json
import ase.io
from pymatgen.io.ase import AseAtomsAdaptor
from mattergen.evaluation.metrics.energy import EnergyMetricsCapability
from mattergen.evaluation.utils.metrics_structure_summary import get_metrics_structure_summaries
from mattergen.evaluation.reference.reference_dataset_serializer import LMDBGZSerializer
from mattergen.evaluation.reference.correction_schemes import TRI110Compatibility2024

base = Path('{out}')
relaxed_atoms = ase.io.read(base / 'relaxed.extxyz', ':')
orig_atoms = ase.io.read(base / 'generated_crystals.extxyz', ':')
structures = [AseAtomsAdaptor.get_structure(x) for x in relaxed_atoms]
orig_structures = [AseAtomsAdaptor.get_structure(x) for x in orig_atoms]
energies = [float(x.info['total_energy']) for x in relaxed_atoms]
reference = LMDBGZSerializer().deserialize('{REFERENCE_DATASET}')
ss = get_metrics_structure_summaries(
    structures=structures, energies=energies,
    original_structures=orig_structures,
    energy_correction_scheme=TRI110Compatibility2024(),
)
cap = EnergyMetricsCapability(structure_summaries=ss, reference_dataset=reference)
summary = {{
    'stable_mask': cap.is_stable.astype(int).tolist(),
    'stable_fraction': float(cap.is_stable.mean()),
    'avg_energy_above_hull_per_atom': float(cap.energy_above_hull.mean()),
}}
(base / 'energy_summary.json').write_text(json.dumps(summary, indent=2))
print(json.dumps(summary))
"""
    script_path = out / "_compute_energy.py"
    script_path.write_text(script.strip())
    cmd = f"PYTHONPATH={MATTERGEN_ROOT} {MATTERSIM_PYTHON} -u {script_path}"
    result = run_cmd(cmd, timeout=300, desc="energy summary")
    return result.returncode == 0


def compute_summary_v2(arm: ExperimentArm) -> dict | None:
    """Combine MatGL scores + energy summary into final summary_v2.json."""
    out = arm.output_dir
    try:
        gen = json.loads((out / "matgl_generated.json").read_text())
        rel = json.loads((out / "matgl_relaxed.json").read_text())
        energy = json.loads((out / "energy_summary.json").read_text())
    except FileNotFoundError as e:
        log(f"  ERROR: missing file for summary — {e}")
        return None

    stable = energy["stable_mask"]
    gen_hits = [1 if x else 0 for x in gen["in_window"]]
    rel_hits = [1 if x else 0 for x in rel["in_window"]]
    n = len(stable)

    summary = {
        "name": arm.name,
        "gamma": arm.gamma,
        "seed": arm.seed,
        "self_correct": arm.self_correct,
        "num_samples": n,
        "generated_hit_rate": sum(gen_hits) / n,
        "relaxed_hit_rate": sum(rel_hits) / n,
        "stable_fraction": energy["stable_fraction"],
        "stable_generated_in_window": sum(s and g for s, g in zip(stable, gen_hits)) / n,
        "stable_relaxed_in_window": sum(s and r for s, r in zip(stable, rel_hits)) / n,
        "avg_energy_above_hull_per_atom": energy["avg_energy_above_hull_per_atom"],
    }

    # Load generation timing if available
    meta_path = out / "generation_meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        summary["generation_time_s"] = meta.get("generation_time_s")

    (out / "summary_v2.json").write_text(json.dumps(summary, indent=2))
    return summary


def run_full_pipeline(arm: ExperimentArm) -> dict | None:
    """Run the complete pipeline for one experiment arm."""
    log(f"{'='*60}")
    log(f"ARM: {arm.name} | gamma={arm.gamma} seed={arm.seed} sc={arm.self_correct}")
    log(f"  output: {arm.output_dir}")

    # Step 1: Generate
    if not generate(arm):
        log(f"  FAILED at generation")
        return None

    # Step 2: MatGL on generated structures
    if not score_matgl(
        arm.output_dir / "generated_crystals.extxyz",
        arm.output_dir / "matgl_generated.json",
        desc="generated",
    ):
        log(f"  FAILED at MatGL generated scoring")
        return None

    # Step 3: MatterSim relaxation + evaluation
    if not evaluate_mattersim(arm):
        log(f"  FAILED at MatterSim evaluation")
        return None

    # Step 4: MatGL on relaxed structures
    if not score_matgl(
        arm.output_dir / "relaxed.extxyz",
        arm.output_dir / "matgl_relaxed.json",
        desc="relaxed",
    ):
        log(f"  FAILED at MatGL relaxed scoring")
        return None

    # Step 5: Compute energy summary (stable_mask, hull)
    if not compute_energy_summary(arm):
        log(f"  FAILED at energy summary computation")
        return None

    # Step 6: Combine into summary_v2.json
    summary = compute_summary_v2(arm)
    if summary:
        log(f"  RESULT: gen_hit={summary['generated_hit_rate']:.3f} "
            f"rel_hit={summary['relaxed_hit_rate']:.3f} "
            f"stable={summary['stable_fraction']:.3f} "
            f"stable+inwindow={summary['stable_relaxed_in_window']:.3f}")
    return summary


def import_existing_v2_results() -> list[dict]:
    """Import existing v2 results (gamma=1.0, seed=2026) into the frontier summary."""
    existing = []
    for name, is_sc in [("scale32_v2_baseline", False), ("scale32_v2_self_correct", True)]:
        path = RESULTS_ROOT / name / "summary_v2.json"
        if path.exists():
            data = json.loads(path.read_text())
            data.update({
                "name": name,
                "gamma": 1.0,
                "seed": 2026,
                "self_correct": is_sc,
                "num_samples": 32,
            })
            # Load generation timing from meta if available
            meta_path = RESULTS_ROOT / name / "generation_meta.json"
            if meta_path.exists():
                meta = json.loads(meta_path.read_text())
                data["generation_time_s"] = meta.get("generation_time_s")
            existing.append(data)
            log(f"Imported existing: {name}")
    return existing


def main():
    log("=" * 60)
    log("MatterGen Self-Correcting Search — Full Frontier Sweep")
    log("=" * 60)

    all_results = []

    log("\n── Importing existing v2 results (gamma=1.0, seed=2026) ──")
    all_results.extend(import_existing_v2_results())

    log("\n── PHASE 1: Cross-Gamma Robustness (gamma=2.0, seed=2026) ──")
    phase1_arms = [
        ExperimentArm(name="frontier_g2.0_s2026_base", gamma=2.0, seed=2026, self_correct=False),
        ExperimentArm(name="frontier_g2.0_s2026_sc", gamma=2.0, seed=2026, self_correct=True),
    ]
    for arm in phase1_arms:
        result = run_full_pipeline(arm)
        if result:
            all_results.append(result)

    log("\n── PHASE 2: Multi-Seed (gamma=1.0, seeds 42 & 137) ──")
    phase2_arms = [
        ExperimentArm(name="frontier_g1.0_s42_base", gamma=1.0, seed=42, self_correct=False),
        ExperimentArm(name="frontier_g1.0_s42_sc", gamma=1.0, seed=42, self_correct=True),
        ExperimentArm(name="frontier_g1.0_s137_base", gamma=1.0, seed=137, self_correct=False),
        ExperimentArm(name="frontier_g1.0_s137_sc", gamma=1.0, seed=137, self_correct=True),
    ]
    for arm in phase2_arms:
        result = run_full_pipeline(arm)
        if result:
            all_results.append(result)

    log("\n── PHASE 3: Pareto Frontier (gamma=0.5, 1.5, 3.0, seed=2026) ──")
    phase3_arms = [
        ExperimentArm(name="frontier_g0.5_s2026_base", gamma=0.5, seed=2026, self_correct=False),
        ExperimentArm(name="frontier_g0.5_s2026_sc", gamma=0.5, seed=2026, self_correct=True),
        ExperimentArm(name="frontier_g1.5_s2026_base", gamma=1.5, seed=2026, self_correct=False),
        ExperimentArm(name="frontier_g1.5_s2026_sc", gamma=1.5, seed=2026, self_correct=True),
        ExperimentArm(name="frontier_g3.0_s2026_base", gamma=3.0, seed=2026, self_correct=False),
        ExperimentArm(name="frontier_g3.0_s2026_sc", gamma=3.0, seed=2026, self_correct=True),
    ]
    for arm in phase3_arms:
        result = run_full_pipeline(arm)
        if result:
            all_results.append(result)

    log("\n── AGGREGATING FRONTIER RESULTS ──")
    frontier_path = RESULTS_ROOT / "frontier_summary.json"
    frontier_path.write_text(json.dumps(all_results, indent=2))
    log(f"Wrote {len(all_results)} arms to {frontier_path}")

    log("\n" + "=" * 100)
    log(f"{'Name':<35} {'gamma':>5} {'seed':>5} {'SC':>3} "
        f"{'gen_hit':>8} {'rel_hit':>8} {'stable':>7} {'s+iw':>7} {'ehull':>7} {'time':>7}")
    log("-" * 100)
    for r in sorted(all_results, key=lambda x: (x["gamma"], x["seed"], x["self_correct"])):
        t = r.get("generation_time_s")
        t_str = f"{t:.0f}s" if t else "n/a"
        log(f"{r['name']:<35} {r['gamma']:>5.1f} {r['seed']:>5} "
            f"{'Y' if r['self_correct'] else 'N':>3} "
            f"{r['generated_hit_rate']:>8.3f} {r['relaxed_hit_rate']:>8.3f} "
            f"{r['stable_fraction']:>7.3f} {r['stable_relaxed_in_window']:>7.3f} "
            f"{r['avg_energy_above_hull_per_atom']:>7.4f} {t_str:>7}")
    log("=" * 100)
    log("SWEEP COMPLETE")


if __name__ == "__main__":
    main()
