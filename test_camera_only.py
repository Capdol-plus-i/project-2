#!/usr/bin/env python3
"""
Test camera functionality only
"""

import os
import cv2

def test_cameras():
    """Test camera connections"""
    print("🎥 Testing camera connections...")

    # Test camera indices
    working_cameras = []
    for i in range(4):
        print(f"Testing camera {i}...")
        cap = cv2.VideoCapture(i, cv2.CAP_V4L2)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                h, w = frame.shape[:2]
                print(f"  ✅ Camera {i}: Working ({w}x{h})")
                working_cameras.append(i)
            else:
                print(f"  ❌ Camera {i}: Can't read frames")
            cap.release()
        else:
            print(f"  ❌ Camera {i}: Can't open")

    print(f"\n📊 Working cameras: {working_cameras}")

    # Test MediaPipe
    print("\n🤖 Testing MediaPipe...")
    try:
        import mediapipe as mp
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.5
        )
        print("✅ MediaPipe hands initialized successfully")

        # Test with a working camera
        if working_cameras:
            print(f"🧪 Testing hand detection with camera {working_cameras[0]}...")
            cap = cv2.VideoCapture(working_cameras[0], cv2.CAP_V4L2)

            for i in range(5):
                ret, frame = cap.read()
                if ret:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = hands.process(rgb)

                    if results.multi_hand_landmarks:
                        for hand_landmarks in results.multi_hand_landmarks:
                            index_tip = hand_landmarks.landmark[8]
                            h, w, _ = frame.shape
                            x, y = int(index_tip.x * w), int(index_tip.y * h)
                            print(f"  📍 Hand detected - Index finger: ({x}, {y})")
                    else:
                        print(f"  👋 No hand detected in frame {i+1}")
                else:
                    print(f"  ❌ Failed to read frame {i+1}")

            cap.release()

    except ImportError:
        print("❌ MediaPipe not available")
    except Exception as e:
        print(f"❌ MediaPipe error: {e}")

if __name__ == "__main__":
    test_cameras()