#!/usr/bin/env bash
set -euo pipefail

: "${EPIC_VIDEO_ROOT:?Set EPIC_VIDEO_ROOT to the EPIC-KITCHENS root that contains participant video folders.}"

OUTPUT_DIR="${OUTPUT_DIR:-ApplyYOLOE/outputs}"
BACKEND="${BACKEND:-fastsam-world}"
DEVICE="${DEVICE:-0}"
PROMPTS="${PROMPTS:-knife cucumber carrot grater hand tap board celery vegetable fork spoon pan onion kettle sponge glass cup plate bowl bottle spatula tofu rice pot sieve}"

mkdir -p "$OUTPUT_DIR"

videos=(
  "P01/P01_01"
  "P01/P01_02"
  "P01/P01_03"
  "P01/P01_04"
  "P01/P01_05"
  "P02/P02_01"
  "P02/P02_03"
  "P10/P10_01"
  "P10/P10_02"
  "P10/P10_03"
)

train_args=()
if [[ -n "${TRAIN_CSV:-}" ]]; then
  train_args=(--train-csv "$TRAIN_CSV")
fi

for item in "${videos[@]}"; do
  participant="${item%%/*}"
  video_id="${item##*/}"
  video_key="$(printf "%s" "$video_id" | tr '[:upper:]' '[:lower:]')"

  python ApplyYOLOE/predict_yoloe_seg_video.py \
    --input "$EPIC_VIDEO_ROOT/$participant/videos/$video_id.MP4" \
    --output "$OUTPUT_DIR/full_$video_id.mp4" \
    --backend "$BACKEND" \
    --device "$DEVICE" \
    --copy-audio \
    --visualize-audio \
    --save-predictions "$OUTPUT_DIR/${video_key}_preds.json" \
    "${train_args[@]}" \
    --prompts $PROMPTS \
    >> "$OUTPUT_DIR/${video_key}_logs.txt"
done
