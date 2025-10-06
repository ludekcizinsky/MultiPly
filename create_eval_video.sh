#!/usr/bin/env bash
set -euo pipefail



# --- paths ---
scene="taichi01"
RENDER_DIR="/scratch/izar/cizinsky/multiply-output/training/$scene/visualisations/test_rendering/-1"
NORMAL_DIR="/scratch/izar/cizinsky/multiply-output/training/$scene/visualisations/test_normal/-1"
FG_DIR="/scratch/izar/cizinsky/multiply-output/training/$scene/visualisations/test_fg_rendering/-1"
OUT_DIR="/scratch/izar/cizinsky/multiply-output/training/$scene/visualisations"

EXT="png"
FPS=10
CRF=18
PRESET="veryfast"

mkdir -p "$OUT_DIR"

ffmpeg -y -hide_banner \
  -framerate "$FPS" -pattern_type glob -i "$RENDER_DIR/*.${EXT}" \
  -framerate "$FPS" -pattern_type glob -i "$NORMAL_DIR/*.${EXT}" \
  -framerate "$FPS" -pattern_type glob -i "$FG_DIR/*.${EXT}" \
  -filter_complex "\
    [0:v]setsar=1[render]; \
    [1:v]setsar=1[normal]; \
    [2:v]setsar=1[fg]; \
    [render][normal][fg]vstack=inputs=3[v]" \
  -map "[v]" -r "$FPS" -c:v libx264 -crf "$CRF" -preset "$PRESET" -pix_fmt yuv420p \
  "$OUT_DIR/stacked.mp4"

echo "✅ Done: $OUT_DIR/stacked.mp4"
