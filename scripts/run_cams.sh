#!/usr/bin/env bash
set -euo pipefail

RX=${1:? "Usage: $0 <receiver_ip> [port_left=5000] [port_right=5002]"}
PL=${2:-5000}
PR=${3:-5002}

L_DEV="/dev/cam_left"
R_DEV="/dev/cam_right"

PIPE() {
  local DEV=$1 PORT=$2 PT=$3
  gst-launch-1.0 -e \
    v4l2src device="$DEV" io-mode=2 ! \
    image/jpeg,width=1280,height=720,framerate=30/1 ! \
    jpegdec ! videoconvert ! queue leaky=downstream max-size-buffers=1 ! \
    x264enc tune=zerolatency speed-preset=ultrafast bitrate=4000 key-int-max=30 bframes=0 ! \
    h264parse config-interval=1 ! rtph264pay pt=$PT mtu=1200 ! \
    udpsink host="10.96.162.204" port="$PORT" sync=false async=false
}

echo "[run_cams] L:$L_DEV -> $RX:$PL(pt=96) | R:$R_DEV -> $RX:$PR(pt=96)"
PIPE "$L_DEV" "$PL" 96 &  # 왼쪽
PIPE "$R_DEV" "$PR" 96 &  # 오른쪽
wait