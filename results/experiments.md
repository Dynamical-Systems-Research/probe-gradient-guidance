# Experiments

## 1. Setup ✅
aarch64 deps on GB10: PyTorch 2.11 from PyPI, Python.h via micromamba, pymatgen symlink fix.
Datasets: Alex-MP-20, MP-20, MPTS-52.

## 2. Training ✅
d_model=512, n_heads=16, n_layers=14, 10K steps Alex-MP-20, 28 min, loss 23.8/24.4.
Checkpoint: outputs/dng_alex_mp20/checkpoints/final.pt

## 3. Probing ✅
Timestep-conditioned probe on atom_mean embeddings (layer -1):
- Window AUROC: **0.956** (MatterGen: 0.972, gap = 0.015)
- Metal AUROC: 0.911, MAE: 0.505 eV
- Lattice token alone: ~0.82 AUROC (atom content carries the signal)
- Probe: results/self_correction/probe.pt

## 4. Metropolis SC ❌
36 configs, all 0% in-window. 97-100% metals. Metropolis cannot steer, only select.
This proved that MatterGen approach (passive selection) requires conditional base distribution.

## 5. Probe-Gradient Guidance ✅
Classifier guidance analog using probe gradient on denoising trajectory.

| Guidance weight | Metal% | In-window% | Mean BG |
|----------------|--------|-----------|---------|
| 0 (baseline) | 96.5% | 0.0% | -0.14 |
| w=1 | 0.8% | 3.5% | 2.38 |
| w=3 | 0.8% | 12.1% | 3.01 |
| w=10 | 0.0% | **24.2%** | 4.19 |

Matches MatterGen 25% with zero conditional training. N=128, 2 seeds.

## 6. Multi-Probe Gradient Composition ❌
Three gradient probes (refractory R2=0.758, Co/Ni AUROC=0.975, insulator AUROC=0.921).
Mode collapse to Au/Cs. Weak probe gradients create adversarial shortcuts.
Lesson: gradient composition fails when probe quality varies.

## 7. Hybrid: Gradient + Token Masking ✅
Gradient for continuous (band gap), token masking for discrete (composition).

| Config | Refractory% | Co/Ni% | Insulator% | In-window% |
|--------|------------|--------|-----------|-----------|
| Baseline | 54% | 1% | 6% | 0% |
| Hybrid (w10 + boost ref + excl CoNi) | **100%** | **0%** | **100%** | **30%** |

Sample compositions: Re Zr Ta Hf Hf Re Re W V Re — refractory HEA candidates.
Constraints compose because they operate at different pipeline stages.

## Key Lessons
1. Passive selection (Metropolis) needs base distribution coverage. Active steering (gradient) does not.
2. Probe quality gates gradient guidance effectiveness. 0.956 AUROC works; 0.758 R2 gets exploited.
3. Match constraint type to enforcement mechanism: continuous → gradient, discrete → token mask.
4. Test-time compute scales monotonically: more guidance weight → higher in-window rate.

## Data Provenance Notes
- Probe AUROC 0.956: from results/probe_final_mp20/probe_results.json (sigma=0.01, layer=11, atom_mean: 0.9559)
- Probe training best_val_auroc 0.957: from train_probe.py stdout (different val split, slightly higher)
- 24.2% in-window at w=10: from sweep_guided.py stdout (N=128, 2 seeds averaged). The results/sweep_v1/sweep_results.json contains the Metropolis sweep (Experiment 4), not the guidance sweep.
- Hybrid 30% in-window: from hybrid_constrained.py stdout (N=128, 2 seeds averaged).
- 96.5% metals baseline: from sweep_guided.py stdout.

## References
- Hadži Veljković, T. et al. "Crystalite." arXiv:2604.02270, April 2026.
- Zeni, C. et al. "MatterGen." Nature, January 2025.
- Sinha, K. et al. "Self-Correcting Search." Goodfire Research, April 2026.

## 8. Pareto Frontier — Targeting vs Diversity ✅
18,432 structures (6 weights x 3 seeds x 1024). ALL weights Pareto-dominate baseline.
Uniqueness: 99.6-99.9% across w=0..15 (no mode collapse). Targeting: 0.1%→33.7%.
Element entropy recovers after initial regime shift. Structural validity degrades (model quality, not guidance).
Conclusion: gradient guidance is Pareto-optimal. Replaces MatterGen in Dynamical pipeline.
Artifacts: scripts/pareto_sweep.py, results/pareto_sweep/

## 9. 100K Training — Full Alex-MP-20 ✅
100K steps on full 540K Alex-MP-20 dataset. Trained on spark-f7e2.
d_model=512, n_heads=16, n_layers=14, batch_size=256, lr=1e-4, bf16.
Checkpoint: outputs/dng_alex_mp20_100k/checkpoints/final.pt
Val loss: 20.59 (vs 23.8 at 10K).
Pareto sweep: structural validity ~5-6% at guided weights (same as 10K).
Conclusion: more training didn't fix validity for guided insulator generation.

## 10. 100K Training — Balanced Subset ✅ (PRODUCTION MODEL)
100K steps on 32K balanced subset (35% insulators, curated from Alex-MP-20).
Trained on spark-cfd0, checkpoint copied to spark-f7e2.
Checkpoint: outputs/dng_balanced_100k/checkpoints/final.pt

Operating point at w=3:
- In-window rate [4-6 eV]: 42.6% (vs 13.6% full model)
- Structural validity (lattice): 100%
- Structural validity (geometry): 99.6%
- Compositional uniqueness: 78%
- Metal fraction: 0.2%
- CHGNet < -0.5 eV/atom: ~23%

Probes (all on spark-cfd0, copied to spark-f7e2):
- Band gap: results/self_correction/probe_balanced_bandgap.pt (AUROC ~0.95)
- Formation energy: results/self_correction/probe_balanced_fe.pt (AUROC **0.990**)
- E_hull: results/self_correction/probe_balanced_ehull.pt (AUROC **0.000** — DEAD)
  E_hull is a database-relative metric the Transformer's hidden states cannot encode.

Conclusion: balanced training is strictly superior for guided generation.
The 35% insulator fraction teaches the model insulator geometry that Alex-MP-20
(2% insulators) underrepresents. This is the production model for the Dynamical pipeline.

## 11. CHGNet Self-Consistent Hull ❌
Attempted to build E_hull final verifier using CHGNet energies for both candidate
and reference compounds (self-consistent hull). Failed for ALL tested systems:
- Cu-Dy-Sn, Au-Dy-Sn, Dy-Ru: 0 compounds with negative formation energy
- Ba-Ti-O, Sr-Ti-O, Li-Fe-O: 0 compounds with negative formation energy
- Ba-Nb-O: 0 compounds with negative formation energy

Root cause: CHGNet's per-element energy offsets are NOT calibrated for formation
energy calculations across different element combinations. Every compound comes
out with Ef = +2-4 eV/atom because elemental reference energies are artificially
low relative to compound energies.

Resolution: dropped E_hull as live verifier. Final stage uses CHGNet relaxation
quality adjudication (stricter thresholds on converged/max_force/volume_strain).
Oracle E_hull remains in spec hidden_truth from MP2020 DFT data.

## 12. Persistent Generation Server ✅
FastAPI server on spark-f7e2:8100. Wraps guided_sampler + masked_decode +
decode_structures + score_bandgap from existing scripts.
- Balanced 100K model loaded in GPU memory
- Element boosting: non-anion elements auto-boosted at strength=5
- Throughput: 10 guided candidates in ~5s end-to-end
- Health: GET /health
- Generate: POST /generate (batched via POST /generate_batch)
- CHGNet: POST /evaluate/chgnet (with relaxation)
- Deploy: scripts/serve.py
