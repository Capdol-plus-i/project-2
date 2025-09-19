#!/usr/bin/env python3
import os, sys, time, argparse, json, socket
import cv2
import numpy as np
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # TF/absl 로그 줄이기

import gi
gi.require_version('Gst', '1.0')                    # ★ 이 줄 추가
from gi.repository import Gst, GObject
# GStreamer 초기화
Gst.init(None)

p = argparse.ArgumentParser(description="MediaPipe 손 오버레이 → RTP(H.264) 송출 (GStreamer/gi)")
p.add_argument("--dev", default=os.environ.get("DEV",
                   "/dev/v4l/by-path/platform-3610000.usb-usb-0:2.4:1.0-video-index0"))
p.add_argument("--rx", required=True)
p.add_argument("--port", type=int, default=int(os.environ.get("PORT", 5001)))
p.add_argument("--pt", type=int, default=int(os.environ.get("PT", 97)))
p.add_argument("--w", type=int, default=int(os.environ.get("W", 640)))
p.add_argument("--h", type=int, default=int(os.environ.get("H", 480)))
p.add_argument("--fps", type=int, default=int(os.environ.get("FPS", 30)))
p.add_argument("--bitrate", type=int, default=int(os.environ.get("BITRATE", 4000)))
p.add_argument("--gop", type=int, default=int(os.environ.get("GOP", 30)))
p.add_argument("--complexity", type=int, default=0)  # MediaPipe model_complexity
p.add_argument("--preview", action="store_true")
p.add_argument("--send-xy", metavar="HOST:PORT",
               help="손 좌표(정규화 0~1) UDP JSON 송신 (예: 127.0.0.1:5555)")
p.add_argument("--no-mp", action="store_true", help="MediaPipe 비활성화 (지연시간 최소화)")
args = p.parse_args()

W, H, FPS = args.w, args.h, args.fps
frame_duration_ns = int(1e9 / FPS)
pts_ns = 0

# ---- 입력: OpenCV로 캡처 (GStreamer 경로 → 실패 시 V4L2) ----
def open_capture():
    pipe_in = (
        f"v4l2src device={args.dev} io-mode=2 ! "
        f"image/jpeg,width={W},height={H},framerate={FPS}/1 ! "
        f"jpegdec ! videoconvert ! video/x-raw,format=BGR ! "
        f"appsink drop=true sync=false"
    )
    cap = cv2.VideoCapture(pipe_in, cv2.CAP_GSTREAMER)
    if cap.isOpened():
        return cap, "gst"
    cap = cv2.VideoCapture(args.dev, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
    cap.set(cv2.CAP_PROP_FPS,          FPS)
    if cap.isOpened():
        return cap, "v4l2"
    return None, None

cap, mode = open_capture()
if not cap:
    print(f"[ERR] 카메라 열기 실패: {args.dev}", file=sys.stderr); sys.exit(1)
print(f"[OK] capture mode = {mode}")

# ---- 출력: GStreamer 파이프라인 (gi) 구성 ----
pipeline_desc = (
    "appsrc name=src is-live=true block=true format=time do-timestamp=true "
    "max-bytes=0 max-buffers=1 ! "
    "queue leaky=downstream max-size-buffers=1 max-size-time=0 max-size-bytes=0 ! "
    "videoconvert ! video/x-raw,format=I420 ! "
    f"x264enc tune=zerolatency speed-preset=ultrafast bitrate={args.bitrate} "
    f"key-int-max={args.gop} bframes=0 ! "
    "h264parse config-interval=1 ! "
    f"rtph264pay pt={args.pt} mtu=1200 ! "
    f"udpsink host={args.rx} port={args.port} sync=false async=false"
)

pipeline = Gst.parse_launch(pipeline_desc)
appsrc = pipeline.get_by_name("src")
caps = Gst.Caps.from_string(f"video/x-raw,format=BGR,width={W},height={H},framerate={FPS}/1")
appsrc.set_property("caps", caps)

ret = pipeline.set_state(Gst.State.PLAYING)
if ret == Gst.StateChangeReturn.FAILURE:
    print("[ERR] GStreamer 파이프라인 시작 실패", file=sys.stderr)
    sys.exit(1)

# ---- MediaPipe 로딩 ----
use_mp = not args.no_mp
if use_mp:
    try:
        import mediapipe as mp
        mp_hands = mp.solutions.hands
        mp_draw = mp.solutions.drawing_utils
        hands = mp_hands.Hands(
            max_num_hands=1,
            model_complexity=args.complexity,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        print("[OK] MediaPipe 활성화")
    except Exception as e:
        use_mp = False
        print("[WARN] MediaPipe 비활성(오버레이 없음):", e)
else:
    print("[INFO] MediaPipe 비활성화 모드 - 지연시간 최소화")

# ---- 좌표 UDP 송신(옵션) ----
sock = None
dst = None
if args.send_xy:
    host, port = args.send_xy.split(":")
    dst = (host, int(port))
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

if args.preview:
    cv2.namedWindow("preview", cv2.WINDOW_NORMAL)

t_prev = time.time()
try:
    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.005); continue

        # MediaPipe 오버레이
        xy_payload = []
        if use_mp:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = hands.process(rgb)
            if res.multi_hand_landmarks:
                for hand in res.multi_hand_landmarks:
                    mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
                    # 좌표 예시: 손목(0), 검지 끝(8) 정규화 좌표
                    wrist = hand.landmark[0]
                    index_tip = hand.landmark[8]
                    xy_payload.append({
                        "wrist": {"x": wrist.x, "y": wrist.y},
                        "index_tip": {"x": index_tip.x, "y": index_tip.y}
                    })

        # FPS 표시
        t_now = time.time()
        fps = 1.0/(t_now - t_prev) if t_now>t_prev else 0.0
        t_prev = t_now
        cv2.putText(frame, f"{fps:4.1f} FPS", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2, cv2.LINE_AA)

        # 좌표 UDP 송신(옵션)
        if sock and xy_payload:
            msg = json.dumps({"ts": time.time(), "hands": xy_payload}).encode("utf-8")
            sock.sendto(msg, dst)

        # GStreamer에 푸시
        # numpy → bytes (연속 메모리 보장)
        data = frame.tobytes() if frame.flags['C_CONTIGUOUS'] else np.ascontiguousarray(frame).tobytes()

        buf = Gst.Buffer.new_allocate(None, len(data), None)
        buf.fill(0, data)
        buf.pts = pts_ns
        buf.dts = pts_ns
        buf.duration = frame_duration_ns
        pts_ns += frame_duration_ns

        flow_ret = appsrc.emit("push-buffer", buf)
        if flow_ret != Gst.FlowReturn.OK:
            print(f"[WARN] push-buffer: {flow_ret}", file=sys.stderr)

        if args.preview:
            cv2.imshow("preview", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

except KeyboardInterrupt:
    pass
finally:
    appsrc.emit("end-of-stream")
    pipeline.set_state(Gst.State.NULL)
    cap.release()
    if args.preview:
        cv2.destroyAllWindows()