#!/usr/bin/env bash
set -e

mkdir -p out

exec python -u hf_color_recognition.py \
  --dataset_root cotton_image \
  --color_db color_dataset.json \
  --segmentation_backend color_threshold \
  --max_size 0 \
  --only_group colorful \
  --allow_classical_fallback \
  --output_json out/debug_color_threshold_colorful_all.json \
  --mask_debug_dir out/debug_masks_color_threshold_colorful_all \
  --overlay_debug_dir out/debug_overlays_color_threshold_colorful_all \
  --log_file out/log_color_threshold_colorful_all.log
