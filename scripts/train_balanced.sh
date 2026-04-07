#!/usr/bin/env bash
# Pipeline: 100K training on balanced Alex-MP-20 subset -> probes -> Pareto sweep
# ~2.5 hours on a single GB10 GPU.
set -euo pipefail
cd ~/crystalite
export PYTHONUNBUFFERED=1

echo "=========================================="
echo "STAGE 1/4: Training 100K steps on balanced subset (~2h)"
echo "Dataset: data/alex_mp20_balanced (32K structures, 35% insulators)"
echo "Started: $(date)"
echo "=========================================="

.venv/bin/python src/train_crystalite.py \
  --data_root data/alex_mp20_balanced --dataset_name alex_mp20 \
  --output_dir outputs/dng_balanced_100k \
  --d_model 512 --n_heads 16 --n_layers 14 \
  --batch_size 256 --max_steps 100000 \
  --sample_frequency 0 \
  --best_ckpt \
  --no_wandb --lr 1e-4 --bf16 \
  --use_distance_bias --use_edge_bias \
  --type_encoding subatomic_tokenizer_pca_24 \
  --ckpt_every 10000 --ckpt_latest_only

echo "=========================================="
echo "STAGE 1 COMPLETE: $(date)"
echo "=========================================="

echo ""
echo "=========================================="
echo "STAGE 2/4: Band-gap probe on balanced model (~3 min)"
echo "Started: $(date)"
echo "=========================================="

.venv/bin/python scripts/train_probe.py \
  --model_checkpoint outputs/dng_balanced_100k/checkpoints/final.pt \
  --data_root data/mp20 \
  --dataset_name mp20 \
  --output_path results/self_correction/probe_balanced_bandgap.pt \
  --n_samples 5000 \
  --probe_layer -1 \
  --hidden_dim 256 \
  --n_epochs 200

echo "=========================================="
echo "STAGE 2 COMPLETE: $(date)"
echo "=========================================="

echo ""
echo "=========================================="
echo "STAGE 3/4: Formation energy probe on balanced model (~3 min)"
echo "Started: $(date)"
echo "=========================================="

# Formation energy probe uses the same script but different window
# Window [low, high] = [-999, -0.5] targets low formation energy (stable)
.venv/bin/python scripts/train_probe.py \
  --model_checkpoint outputs/dng_balanced_100k/checkpoints/final.pt \
  --data_root data/mp20 \
  --dataset_name mp20 \
  --output_path results/self_correction/probe_balanced_fe.pt \
  --n_samples 5000 \
  --probe_layer -1 \
  --hidden_dim 256 \
  --n_epochs 200 \
  --window_low -999.0 \
  --window_high -0.5

echo "=========================================="
echo "STAGE 3 COMPLETE: $(date)"
echo "=========================================="

echo ""
echo "=========================================="
echo "STAGE 4/4: Pareto sweep with balanced model (~30 min)"
echo "Started: $(date)"
echo "=========================================="

.venv/bin/python scripts/pareto_sweep.py \
  --model_checkpoint outputs/dng_balanced_100k/checkpoints/final.pt \
  --probe_path results/self_correction/probe_balanced_bandgap.pt \
  --output_dir results/pareto_sweep_balanced

echo "=========================================="
echo "PIPELINE COMPLETE: $(date)"
echo "Results: results/pareto_sweep_balanced/"
echo "=========================================="

touch results/pareto_sweep_balanced/PIPELINE_DONE
