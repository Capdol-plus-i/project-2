#!/usr/bin/env bash
set -euo pipefail
DEVS=("/dev/cam_left" "/dev/cam_right")

for DEV in "${DEVS[@]}"; do
  [[ -e "$DEV" ]] || continue
  echo "[preset] $DEV"
  v4l2-ctl -d "$DEV" -c contrast=255,saturation=255 || true
done