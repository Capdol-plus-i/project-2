#!/usr/bin/env bash
set -euo pipefail

# ===== 기본 설정 =====
: "${RX:=10.96.162.204}"   # 수신 PC IP (원하는 IP로 바꿔도 되고, 실행 전에 export RX=... 로 덮어써도 됨)

DEV_LEFT="/dev/v4l/by-path/platform-3610000.usb-usb-0:2.2:1.0-video-index0"
DEV_RIGHT="/dev/v4l/by-path/platform-3610000.usb-usb-0:2.4:1.0-video-index0"

PORT_LEFT=5000   PT_LEFT=96
PORT_RIGHT=5002  PT_RIGHT=97

WIDTH=640  HEIGHT=480  FPS=30
BITRATE=4000        # kbps
GOP=30              # key-int-max(프레임)

mkdir -p "$HOME/gst-logs"

run_cam() {
  local DEV="$1" PORT="$2" PT="$3" NAME="$4"
  echo "[run_cams] $NAME: $DEV -> udp://$RX:$PORT (pt=$PT)"
  until [ -e "$DEV" ]; do echo "[run_cams] 대기중: $DEV"; sleep 1; done

  # MJPG → H.264(RTP/UDP)
  gst-launch-1.0 -e \
    v4l2src device="$DEV" io-mode=2 do-timestamp=true ! \
    image/jpeg,width=$WIDTH,height=$HEIGHT,framerate=$FPS/1 ! \
    jpegdec ! videoconvert ! video/x-raw,format=I420 ! \
    queue leaky=downstream max-size-buffers=1 ! \
    x264enc tune=zerolatency speed-preset=ultrafast bitrate=$BITRATE key-int-max=$GOP bframes=0 ! \
    h264parse config-interval=1 ! rtph264pay pt=$PT mtu=1200 ! \
    udpsink host="$RX" port="$PORT" sync=false async=false \
    >"$HOME/gst-logs/$NAME.log" 2>&1 &
  echo $! >"$HOME/gst-logs/$NAME.pid"
}

cleanup() {
  for n in left right; do
    if [ -f "$HOME/gst-logs/$n.pid" ]; then
      kill "$(cat "$HOME/gst-logs/$n.pid")" 2>/dev/null || true
      rm -f "$HOME/gst-logs/$n.pid"
    fi
  done
}
trap cleanup EXIT INT TERM

# 한쪽만 테스트하고 싶으면 ONLY=left 또는 ONLY=right 로 실행
ONLY="${ONLY:-both}"
if [ "$ONLY" = "left" ]; then
  run_cam "$DEV_LEFT" "$PORT_LEFT" "$PT_LEFT" left
elif [ "$ONLY" = "right" ]; then
  run_cam "$DEV_RIGHT" "$PORT_RIGHT" "$PT_RIGHT" right
else
  run_cam "$DEV_LEFT" "$PORT_LEFT" "$PT_LEFT" left
  run_cam "$DEV_RIGHT" "$PORT_RIGHT" "$PT_RIGHT" right
fi

echo "[run_cams] 송신 중. 로그: $HOME/gst-logs/{left,right}.log"
wait
