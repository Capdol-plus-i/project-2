import os, sys, time, argparse
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # TF/absl 로그 줄이기
import cv2

# 명령행 인자 파싱
parser = argparse.ArgumentParser(description="Dual Camera Hand Tracking")
parser.add_argument("--headless", action="store_true", help="Run in headless mode (no display)")
parser.add_argument("--output", help="Output video file path (headless mode only)")
args = parser.parse_args()

# 카메라 설정
DEV1 = os.environ.get("DEV", "/dev/v4l/by-path/platform-3610000.usb-usb-0:2.2:1.0-video-index0")
DEV2 = os.environ.get("DEV2", "/dev/v4l/by-path/platform-3610000.usb-usb-0:2.4:1.0-video-index0")
WIDTH  = int(os.environ.get("W", 640))
HEIGHT = int(os.environ.get("H", 480))
FPS    = int(os.environ.get("FPS", 30))

def open_camera(device_id):
    """카메라 열기 및 최적화 설정"""
    if isinstance(device_id, str) and device_id.startswith('/dev/'):
        # 경로인 경우 숫자로 변환 시도
        try:
            device_id = int(device_id.split('video')[1])
        except:
            device_id = 0

    cap = cv2.VideoCapture(device_id, cv2.CAP_V4L2)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, FPS)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap
    return None

# 두 개의 카메라 초기화
cap1 = open_camera(0)
cap2 = open_camera(2)

if not cap1:
    print("첫 번째 카메라 열기 실패", file=sys.stderr)
    sys.exit(1)

print(f"첫 번째 카메라: OK")
if cap2:
    print(f"두 번째 카메라: OK")
else:
    print(f"두 번째 카메라: 사용 불가")

# MediaPipe (선택): 설치되어 있으면 사용
use_mp = False
hands1 = None
hands2 = None
try:
    import mediapipe as mp
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    # 각 카메라별로 별도의 MediaPipe 인스턴스 생성
    hands1 = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,  # 동시 감지를 위해 증가
        model_complexity=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.5
    )
    hands2 = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        model_complexity=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.5
    )
    use_mp = True
    print("MediaPipe 활성화: 손 랜드마크 추적")
except Exception as e:
    print("MediaPipe 미설치 또는 초기화 실패 – 영상만 표시합니다.", e)

def process_frame(cap, hands_instance, use_mediapipe=True):
    """프레임 처리 및 MediaPipe 적용"""
    ok, frame = cap.read()
    if not ok:
        return None, []

    finger_coords = []
    if use_mediapipe and use_mp and hands_instance:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands_instance.process(rgb)
        if res.multi_hand_landmarks:
            for hand_landmarks in res.multi_hand_landmarks:
                # 랜드마크 그리기
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # 검지 끝 좌표 (랜드마크 8번)
                index_tip = hand_landmarks.landmark[8]
                # 손목 좌표 (랜드마크 0번)
                wrist = hand_landmarks.landmark[0]

                h, w, _ = frame.shape
                idx_x, idx_y = int(index_tip.x * w), int(index_tip.y * h)
                wrist_x, wrist_y = int(wrist.x * w), int(wrist.y * h)

                # 검지 끝에 빨간 원, 손목에 초록 원 그리기
                cv2.circle(frame, (idx_x, idx_y), 8, (0, 0, 255), -1)  # 빨간색
                cv2.circle(frame, (wrist_x, wrist_y), 6, (0, 255, 0), -1)  # 초록색

                # 좌표 저장 (검지, 손목)
                finger_coords.append({
                    'index_tip': (idx_x, idx_y, index_tip.x, index_tip.y),
                    'wrist': (wrist_x, wrist_y, wrist.x, wrist.y)
                })

    return frame, finger_coords

# 디스플레이 설정
if not args.headless:
    win = "Dual Camera Hand Demo"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
else:
    print("Headless 모드로 실행 중...")

# 비디오 출력 설정 (headless + output 옵션)
video_writer = None
if args.headless and args.output:
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_writer = cv2.VideoWriter(args.output, fourcc, FPS, (WIDTH*2 if cap2 else WIDTH, HEIGHT))
    print(f"비디오 출력: {args.output}")

prev = time.time()
frame_count = 0
skip_frames = 1

try:
    while True:
        frame_count += 1
        process_mp = (frame_count % skip_frames == 0)

        # 첫 번째 카메라 처리
        frame1, coords1 = process_frame(cap1, hands1 if use_mp else None, process_mp)
        if frame1 is None:
            time.sleep(0.005)
            continue

        # 두 번째 카메라 처리 (있다면)
        frame2 = None
        coords2 = []
        if cap2:
            frame2, coords2 = process_frame(cap2, hands2 if use_mp else None, process_mp)

        # FPS 계산
        now = time.time()
        dt = now - prev
        if dt > 0:
            fps = 1.0 / dt
        prev = now

        # FPS 및 좌표 정보 표시
        cv2.putText(frame1, f"{fps:4.1f} FPS", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame1, "Camera 1", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        # 첫 번째 카메라 손 좌표 표시
        for i, hand_data in enumerate(coords1):
            idx_x, idx_y, idx_nx, idx_ny = hand_data['index_tip']
            w_x, w_y, w_nx, w_ny = hand_data['wrist']

            text1 = f"C1 H{i+1} Index: ({idx_x},{idx_y})"
            text2 = f"C1 H{i+1} Wrist: ({w_x},{w_y})"
            cv2.putText(frame1, text1, (10, 90 + i*50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.putText(frame1, text2, (10, 110 + i*50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)

        if frame2 is not None:
            cv2.putText(frame2, f"{fps:4.1f} FPS", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame2, "Camera 2", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

            # 두 번째 카메라 손 좌표 표시
            for i, hand_data in enumerate(coords2):
                idx_x, idx_y, idx_nx, idx_ny = hand_data['index_tip']
                w_x, w_y, w_nx, w_ny = hand_data['wrist']

                text1 = f"C2 H{i+1} Index: ({idx_x},{idx_y})"
                text2 = f"C2 H{i+1} Wrist: ({w_x},{w_y})"
                cv2.putText(frame2, text1, (10, 90 + i*50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)
                cv2.putText(frame2, text2, (10, 110 + i*50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)

            # 두 프레임을 나란히 표시
            combined_frame = cv2.hconcat([frame1, frame2])
        else:
            combined_frame = frame1

        # 화면 표시 또는 비디오 저장
        if not args.headless:
            cv2.imshow(win, combined_frame)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC
                break
        else:
            # headless 모드에서는 좌표 정보를 콘솔에 출력
            if coords1:
                for i, hand_data in enumerate(coords1):
                    idx_x, idx_y, _, _ = hand_data['index_tip']
                    w_x, w_y, _, _ = hand_data['wrist']
                    print(f"C1 H{i+1} - Index:({idx_x},{idx_y}) Wrist:({w_x},{w_y})")

            if coords2:
                for i, hand_data in enumerate(coords2):
                    idx_x, idx_y, _, _ = hand_data['index_tip']
                    w_x, w_y, _, _ = hand_data['wrist']
                    print(f"C2 H{i+1} - Index:({idx_x},{idx_y}) Wrist:({w_x},{w_y})")

            # 비디오 파일 저장
            if video_writer:
                video_writer.write(combined_frame)

            # headless 모드에서는 Ctrl+C로만 종료 가능
            time.sleep(0.01)

except KeyboardInterrupt:
    print("\n프로그램 종료")

finally:
    cap1.release()
    if cap2:
        cap2.release()
    if video_writer:
        video_writer.release()
        print(f"비디오 저장 완료: {args.output}")
    if not args.headless:
        cv2.destroyAllWindows()