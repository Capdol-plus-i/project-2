#!/usr/bin/env python3
import os, sys, time, argparse, json, socket
import cv2
import numpy as np
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # TF/absl 로그 줄이기

p = argparse.ArgumentParser(description="MediaPipe 손 오버레이 (OpenCV)")
p.add_argument("--dev", default=os.environ.get("DEV",
                   "/dev/v4l/by-path/platform-3610000.usb-usb-0:2.2:1.0-video-index0"))
p.add_argument("--dev2", default=os.environ.get("DEV2",
                   "/dev/v4l/by-path/platform-3610000.usb-usb-0:2.4:1.0-video-index0"))
p.add_argument("--w", type=int, default=int(os.environ.get("W", 640)))
p.add_argument("--h", type=int, default=int(os.environ.get("H", 480)))
p.add_argument("--fps", type=int, default=int(os.environ.get("FPS", 30)))
p.add_argument("--complexity", type=int, default=0)  # MediaPipe model_complexity
p.add_argument("--preview", action="store_true")
p.add_argument("--send-xy", metavar="HOST:PORT",
               help="손 좌표(정규화 0~1) UDP JSON 송신 (예: 127.0.0.1:5555)")
p.add_argument("--no-mp", action="store_true", help="MediaPipe 비활성화 (지연시간 최소화)")
args = p.parse_args()

W, H, FPS = args.w, args.h, args.fps

# ---- 입력: OpenCV로 캡처 ----
def open_capture(device):
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
    cap.set(cv2.CAP_PROP_FPS,          FPS)
    if cap.isOpened():
        return cap, "v4l2"
    return None, None

cap1, mode1 = open_capture(args.dev)
if not cap1:
    print(f"[ERR] 첫 번째 카메라 열기 실패: {args.dev}", file=sys.stderr); sys.exit(1)
print(f"[OK] 첫 번째 카메라 mode = {mode1}")

cap2 = None
mode2 = None
if args.dev2:
    cap2, mode2 = open_capture(args.dev2)
    if cap2:
        print(f"[OK] 두 번째 카메라 mode = {mode2}")
    else:
        print(f"[WARN] 두 번째 카메라 열기 실패: {args.dev2}")
else:
    print("[INFO] 두 번째 카메라 비활성화")

# ---- MediaPipe 로딩 ----
use_mp = not args.no_mp
if use_mp:
    try:
        import mediapipe as mp
        mp_hands = mp.solutions.hands
        mp_draw = mp.solutions.drawing_utils
        hands = mp_hands.Hands(
            max_num_hands=1,
            model_complexity=0,
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
    cv2.namedWindow("카메라 피드", cv2.WINDOW_NORMAL)

def process_frame(cap, window_name):
    ok, frame = cap.read()
    if not ok:
        return None, []

    xy_payload = []
    if use_mp:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)
        if res.multi_hand_landmarks:
            for hand in res.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
                wrist = hand.landmark[0]
                index_tip = hand.landmark[8]
                xy_payload.append({
                    "camera": window_name,
                    "wrist": {"x": wrist.x, "y": wrist.y},
                    "index_tip": {"x": index_tip.x, "y": index_tip.y}
                })

    return frame, xy_payload

t_prev = time.time()
try:
    while True:
        frame1, xy_payload1 = process_frame(cap1, "왼쪽")
        if frame1 is None:
            time.sleep(0.005); continue

        frame2, xy_payload2 = None, []
        if cap2:
            frame2, xy_payload2 = process_frame(cap2, "오른쪽")

        # FPS 표시
        t_now = time.time()
        fps = 1.0/(t_now - t_prev) if t_now>t_prev else 0.0
        t_prev = t_now

        cv2.putText(frame1, f"{fps:4.1f} FPS", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2, cv2.LINE_AA)
        if frame2 is not None:
            cv2.putText(frame2, f"{fps:4.1f} FPS", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2, cv2.LINE_AA)

        # 좌표 UDP 송신(옵션)
        all_payload = xy_payload1 + xy_payload2
        if sock and all_payload:
            msg = json.dumps({"ts": time.time(), "hands": all_payload}).encode("utf-8")
            sock.sendto(msg, dst)

        if args.preview:
            if frame2 is not None:
                # 두 프레임을 수평으로 연결
                combined_frame = np.hstack((frame1, frame2))
            else:
                combined_frame = frame1

            cv2.imshow("카메라 피드", combined_frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

except KeyboardInterrupt:
    pass
finally:
    cap1.release()
    if cap2:
        cap2.release()
    if args.preview:
        cv2.destroyAllWindows()