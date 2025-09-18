import os, sys, time
import cv2

# 카메라 by-path 기본값(왼쪽)
DEV = os.environ.get("DEV", "/dev/v4l/by-path/platform-3610000.usb-usb-0:2.2:1.0-video-index0")
WIDTH  = int(os.environ.get("W", 1280))
HEIGHT = int(os.environ.get("H", 720))
FPS    = int(os.environ.get("FPS", 30))

# GStreamer 캡쳐(카메라가 MJPG일 때)
pipeline = (
    f'v4l2src device={DEV} io-mode=2 '
    f'! image/jpeg,width={WIDTH},height={HEIGHT},framerate={FPS}/1 '
    f'! jpegdec ! videoconvert '
    f'! appsink drop=true sync=false'
)

cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
if not cap.isOpened():
    print("카메라 열기 실패:", DEV, file=sys.stderr); sys.exit(1)

# MediaPipe (선택): 설치되어 있으면 사용
use_mp = False
try:
    import mediapipe as mp
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        max_num_hands=2,
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    use_mp = True
    print("MediaPipe 활성화: 손 랜드마크 추적")
except Exception as e:
    print("MediaPipe 미설치 또는 초기화 실패 – 영상만 표시합니다.", e)

win = "Hand Demo"
cv2.namedWindow(win, cv2.WINDOW_NORMAL)

prev = time.time()
while True:
    ok, frame = cap.read()
    if not ok: break
    if use_mp:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)
        if res.multi_hand_landmarks:
            for h in res.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, h, mp_hands.HAND_CONNECTIONS)
    # FPS 표기
    now = time.time()
    fps = 1.0/(now-prev) if now>prev else 0.0
    prev = now
    cv2.putText(frame, f"{fps:4.1f} FPS", (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2, cv2.LINE_AA)
    cv2.imshow(win, frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()