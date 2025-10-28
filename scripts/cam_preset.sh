#!/usr/bin/env bash
set -euo pipefail

# Camera device definitions
LEFT_CAM="/dev/cam_left"
RIGHT_CAM="/dev/cam_right"

# Common settings for both cameras
CONTRAST=255
SATURATION=255
GAMMA=30
SHARPNESS=0

# Different brightness settings
LEFT_BRIGHTNESS=-200    # Brighter for left camera
RIGHT_BRIGHTNESS=-150 # Keep right camera darker

# Configure left camera
if [[ -e "$LEFT_CAM" ]]; then
  echo "[preset] Setting $LEFT_CAM with increased brightness"

  v4l2-ctl -d "$LEFT_CAM" --set-ctrl=brightness=$LEFT_BRIGHTNESS || true
  v4l2-ctl -d "$LEFT_CAM" --set-ctrl=contrast=$CONTRAST || true
  v4l2-ctl -d "$LEFT_CAM" --set-ctrl=saturation=$SATURATION || true
  v4l2-ctl -d "$LEFT_CAM" --set-ctrl=gamma=$GAMMA || true
  v4l2-ctl -d "$LEFT_CAM" --set-ctrl=sharpness=$SHARPNESS || true

  echo "[preset] Left camera brightness set to $LEFT_BRIGHTNESS"
fi

# Configure right camera
if [[ -e "$RIGHT_CAM" ]]; then
  echo "[preset] Setting $RIGHT_CAM with original brightness"

  v4l2-ctl -d "$RIGHT_CAM" --set-ctrl=brightness=$RIGHT_BRIGHTNESS || true
  v4l2-ctl -d "$RIGHT_CAM" --set-ctrl=contrast=$CONTRAST || true
  v4l2-ctl -d "$RIGHT_CAM" --set-ctrl=saturation=$SATURATION || true
  v4l2-ctl -d "$RIGHT_CAM" --set-ctrl=gamma=$GAMMA || true
  v4l2-ctl -d "$RIGHT_CAM" --set-ctrl=sharpness=$SHARPNESS || true

  echo "[preset] Right camera brightness set to $RIGHT_BRIGHTNESS"
fi

echo "[preset] Camera settings applied - Left: brighter ($LEFT_BRIGHTNESS), Right: darker ($RIGHT_BRIGHTNESS)"
