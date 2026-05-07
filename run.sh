#!/usr/bin/env bash
set -euo pipefail

mkdir -p out

LOG_FILE="out/log_colorful_vote_soft_all.log"

exec > >(tee -a "$LOG_FILE") 2>&1

exec python -u main.py \
  --dataset_root cotton_image \
  --color_db_path color_dataset.json \
  --only_group colorful \
  --dominant_method vote \
  --stabilize_outputs \
  --stabilization_mode soft \
  --stabilize_alpha 0.25 \
  --l_bin 2.0 \
  --ab_bin 2.0 \
  --l_trim_low 5 \
  --l_trim_high 95 \
  --max_vote_pixels 300000 \
  --output out/colorful_vote_soft_all_final.json \
  --mask_debug_dir out/debug_masks_colorful_all \
  --overlay_debug_dir out/debug_overlays_colorful_all