import os
import sys
import cv2
try:
    import mediapipe as mp
except Exception as e:
    msg = str(e)
    if "MessageFactory" in msg and "GetPrototype" in msg:
        print(
            "mediapipe/protobuf version mismatch detected.\n"
            "Fix: pin protobuf to 3.20.3 (or <4). For example:\n"
            "  pip install --upgrade 'protobuf==3.20.3'\n"
            "Optionally ensure mediapipe < 0.11 if needed."
        )
    else:
        print(f"Failed to import mediapipe: {e}")
    sys.exit(1)

# MediaPipe Hands 솔루션 초기화
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)
mp_drawing = mp.solutions.drawing_utils

def open_first_camera(preferred=(0, 1, 2, 3, 4)):
    backends = []
    # Prefer V4L2 on Linux
    if os.name == 'posix':
        backends = [cv2.CAP_V4L2, cv2.CAP_ANY]
    else:
        backends = [cv2.CAP_ANY]

    for idx in preferred:
        for be in backends:
            cap = cv2.VideoCapture(idx, be)
            if not cap.isOpened():
                cap.release()
                continue
            # sanity read
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            ok, frame = cap.read()
            if ok and frame is not None and frame.size > 0:
                print(f"Using camera index {idx} (backend {be})")
                return cap
            cap.release()
    return None

# 웹캠 열기 (자동 탐지)
cap = open_first_camera()
if cap is None:
    print("카메라를 열 수 없습니다. 다른 인덱스/권한/점유 상태를 확인하세요.")
    if os.name == 'posix':
        # 간단한 힌트 제공 (사용자 시스템에서 실행 필요)
        print("- 확인: ls /dev/video*  • v4l2-ctl --list-devices")
        print("- 점유: fuser /dev/video0  • 권한: 사용자 video 그룹 여부")
    sys.exit(1)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("카메라 프레임을 읽을 수 없습니다.")
        continue

    # 성능 향상을 위해 이미지를 BGR에서 RGB로 변환
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 이미지 처리 및 손 랜드마크 검출
    results = hands.process(image_rgb)

    # 좌우 반전 (거울 모드)
    image = cv2.flip(image, 1)

    # 손이 검출되었을 경우
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # 손목(WRIST, id=0)의 좌표 추출
            # 좌표는 0.0 ~ 1.0 사이의 값으로 정규화되어 있음
            wrist_landmark = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
            
            # 이미지의 너비와 높이
            h, w, c = image.shape
            
            # 실제 픽셀 좌표 계산
            cx, cy = int(wrist_landmark.x * w), int(wrist_landmark.y * h)
            
            # 터미널에 좌표 출력 (이 값을 로봇 제어에 사용)
            print(f"손목 좌표 (x, y): ({cx}, {cy})")

            # 화면에 랜드마크 그리기
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS)
            
            # 손목 위치에 원 그리기
            cv2.circle(image, (cx, cy), 10, (255, 0, 255), -1)


    # 자원 해제
cap.release()