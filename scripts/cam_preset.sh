#!/usr/bin/env bash
set -euo pipefail

# Camera device definitions
LEFT_CAM="/dev/video0"
RIGHT_CAM="/dev/video2"

# Common settings for both cameras
CONTRAST=255
SATURATION=255
GAMMA=35
GAIN=0
SHARPNESS=0
POWER_LINE_FREQ=2  # 60 Hz
AUTO_EXPOSURE=3    # Aperture Priority Mode
FOCUS_AUTO=1       # Auto focus enabled

# Different brightness settings
LEFT_BRIGHTNESS=-150    # Brighter for left camera
RIGHT_BRIGHTNESS=-200 # Keep right camera darker

# Configure left camera
if [[ -e "$LEFT_CAM" ]]; then
  echo "[preset] Setting $LEFT_CAM with increased brightness"

  v4l2-ctl -d "$LEFT_CAM" --set-ctrl=brightness=$LEFT_BRIGHTNESS || true
  v4l2-ctl -d "$LEFT_CAM" --set-ctrl=contrast=$CONTRAST || true
  v4l2-ctl -d "$LEFT_CAM" --set-ctrl=saturation=$SATURATION || true
  v4l2-ctl -d "$LEFT_CAM" --set-ctrl=gamma=$GAMMA || true
  v4l2-ctl -d "$LEFT_CAM" --set-ctrl=gain=$GAIN || true
  v4l2-ctl -d "$LEFT_CAM" --set-ctrl=sharpness=$SHARPNESS || true
  v4l2-ctl -d "$LEFT_CAM" --set-ctrl=power_line_frequency=$POWER_LINE_FREQ || true
  v4l2-ctl -d "$LEFT_CAM" --set-ctrl=auto_exposure=$AUTO_EXPOSURE || true
  v4l2-ctl -d "$LEFT_CAM" --set-ctrl=focus_automatic_continuous=$FOCUS_AUTO || true

  echo "[preset] Left camera brightness set to $LEFT_BRIGHTNESS"
fi

# Configure right camera
if [[ -e "$RIGHT_CAM" ]]; then
  echo "[preset] Setting $RIGHT_CAM with original brightness"

  v4l2-ctl -d "$RIGHT_CAM" --set-ctrl=brightness=$RIGHT_BRIGHTNESS || true
  v4l2-ctl -d "$RIGHT_CAM" --set-ctrl=contrast=$CONTRAST || true
  v4l2-ctl -d "$RIGHT_CAM" --set-ctrl=saturation=$SATURATION || true
  v4l2-ctl -d "$RIGHT_CAM" --set-ctrl=gamma=$GAMMA || true
  v4l2-ctl -d "$RIGHT_CAM" --set-ctrl=gain=$GAIN || true
  v4l2-ctl -d "$RIGHT_CAM" --set-ctrl=sharpness=$SHARPNESS || true
  v4l2-ctl -d "$RIGHT_CAM" --set-ctrl=power_line_frequency=$POWER_LINE_FREQ || true
  v4l2-ctl -d "$RIGHT_CAM" --set-ctrl=auto_exposure=$AUTO_EXPOSURE || true
  v4l2-ctl -d "$RIGHT_CAM" --set-ctrl=focus_automatic_continuous=$FOCUS_AUTO || true

  echo "[preset] Right camera brightness set to $RIGHT_BRIGHTNESS"
fi

echo "[preset] Camera settings applied - Left: brighter ($LEFT_BRIGHTNESS), Right: darker ($RIGHT_BRIGHTNESS)"