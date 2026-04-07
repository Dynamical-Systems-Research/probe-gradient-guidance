# Probe-Gradient Guidance

Test-time verification for crystal structure generation. A 256-parameter linear probe steers an unconditional diffusion model toward target material properties by backpropagating through the probe at each denoising step. No retraining. No conditional model. Swap the probe, change the target.

![Pareto targeting: in-window rate climbs from 0.1% to 33.7% across guidance weights with no loss in compositional uniqueness (99.6-99.9%)](assets/pareto-targeting.png)

**Blog post**: [Scaling Test-Time Verification for Novel Materials](https://dynamicalsystems.ai/blog/scaling-test-time-verification)

## Results

On [Crystalite](https://arxiv.org/abs/2604.02270) (67.8M-parameter Diffusion Transformer, trained on Alex-MP-20):

| Guidance weight | Metal % | In-window (4-6 eV) % | Mean band gap (eV) |
|---|---|---|---|
| 0 (baseline) | 96.5 | 0.0 | -0.14 |
| 1 | 0.8 | 3.5 | 2.38 |
| 3 | 0.8 | 12.1 | 3.01 |
| 10 | 0.0 | 24.2 | 4.19 |

24.2% in-window from an unconditional model trained on 97.9% metals. Matches [MatterGen](https://www.nature.com/articles/s41586-025-08628-5) conditional generation with [self-correcting search](https://www.goodfire.ai/research/self-correcting-search) (25-28%) at ~500x the speed.

**Balanced model** (32K subset, 35% insulators): 42.6% in-window, 100% lattice validity, 99.6% geometry validity. Formation energy probe AUROC: 0.990.

## Setup

```bash
git clone https://github.com/Dynamical-Systems-Research/probe-gradient-guidance.git
cd probe-gradient-guidance
pip install -r requirements.txt
```

Requires the [Crystalite](https://arxiv.org/abs/2604.02270) codebase installed at the repo root as `src/`. See the Crystalite paper for architecture details and training code.

## Quick start

```bash
# Train a probe on model hidden states
python scripts/train_probe.py \
    --model_checkpoint outputs/dng_alex_mp20/checkpoints/final.pt \
    --output_path probes/bandgap_10k.pt

# Full Pareto sweep (18K structures, 6 weights, 3 seeds)
python scripts/pareto.py \
    --model_checkpoint outputs/dng_alex_mp20/checkpoints/final.pt \
    --probe_path probes/bandgap_10k.pt \
    --output_dir results/pareto_sweep
```

`scripts/sweep.py`, `scripts/constrained.py`, and `scripts/generate.py` are research scripts with hardcoded paths at the top of each file. Edit the path constants before running. `scripts/serve.py` is a FastAPI server with full argparse.

## Trained probes

| Probe | Model | Property | AUROC |
|---|---|---|---|
| `probes/bandgap_10k.pt` | Crystalite 10K | Band gap | 0.957 |
| `probes/bandgap_balanced.pt` | Crystalite balanced | Band gap | ~0.95 |
| `probes/formation_energy_balanced.pt` | Crystalite balanced | Formation energy | 0.990 |
| `probes/bandgap_mattergen.pt` | MatterGen | Band gap | 0.972 |

## Model checkpoints

Crystalite checkpoints (519MB each) are hosted on HuggingFace:

- [`Dynamical-Systems/crystalite-10k-alex-mp20`](https://huggingface.co/Dynamical-Systems/crystalite-10k-alex-mp20) (diversity-optimized)
- [`Dynamical-Systems/crystalite-balanced-100k`](https://huggingface.co/Dynamical-Systems/crystalite-balanced-100k) (production, 42.6% in-window)

Requires the [Crystalite](https://arxiv.org/abs/2604.02270) codebase at `src/` for model architecture.

## Repo structure

```
scripts/
  generate.py          Probe-gradient guidance sampler
  train_probe.py       Probe training
  sweep.py             Guidance weight sweep
  pareto.py            18K structure Pareto sweep (6 weights x 3 seeds x 1024)
  constrained.py       Multi-constraint: gradient steering + token masking
  metropolis.py        Metropolis accept/reject baseline
  evaluate.py          Probe + CHGNet evaluation pipeline
  serve.py             FastAPI generation server
  train_balanced.sh    Balanced training configuration

mattergen_repro/
  frontier_v2.py       MatterGen SC reproduction (Goodfire, v2)
  frontier_v3.py       MatterGen SC reproduction (v3, best-of-3)
  sampler_patch.py     MatterGen sampler patch

probes/                Trained probe checkpoints
results/               Reproducibility data (Pareto sweep, MatterGen frontier)
```

## References

- Hadzi Veljkovic, T. et al. [Crystalite: A Lightweight Transformer for Efficient Crystal Modeling](https://arxiv.org/abs/2604.02270). arXiv:2604.02270, 2026.
- Sinha, K. et al. [Using Self-Correcting Search to Accelerate Materials Discovery](https://www.goodfire.ai/research/self-correcting-search). Goodfire Research, 2026.
- Zeni, C. et al. [A Generative Model for Inorganic Materials Design](https://www.nature.com/articles/s41586-025-08628-5). Nature, 2025.

## Citation

```bibtex
@article{barnes2026verification,
  author  = {Barnes, Jarrod},
  title   = {Scaling Test-Time Verification for Novel Materials},
  journal = {Dynamical Systems},
  year    = {2026},
  url     = {https://dynamicalsystems.ai/blog/scaling-test-time-verification}
}
```

## License

MIT
