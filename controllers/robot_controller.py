#!/usr/bin/env python3
"""
Robot Controller module for camera processing and hand gesture recognition.
"""

import os
import time
import threading
import logging
import numpy as np
import cv2
import mediapipe as mp
from config import CAM_IDS, ATTEMPTS_READ_FRAME_COUNT, DEFAULT_JOINTS, HANDS_TIMEOUT

logger = logging.getLogger(__name__)
mp_hands = mp.solutions.hands

class RobotController:
    def __init__(self, robot, model_path=None, arm_type='follower'):
        self.robot = robot
        self.arm_type = arm_type
        self.cams = [None, None]
        self.width, self.height = 640, 480
        self.running = False
        self.control_active = False
        self.data_lock = threading.Lock()
        self.last_frames = [None, None]
        self.last_data = [None] * 8
        self.tip = [(0,0), (0,0)]
        self.hand_detected = [False, False]
        self.z = 10
        self.last_status_update = 0
        self.status_update_interval = 0.1
        
        # Load model
        self.params = None
        if model_path and os.path.exists(model_path):
            try:
                params = np.load(model_path)
                self.params = {k: params[k] for k in params.files}
                logger.info(f"Model loaded from {model_path}")
            except Exception as e:
                logger.error(f"Model load error: {e}")
        else:
            logger.warning(f"Model not found at {model_path}, using dummy predictions")
        
        # Initialize MediaPipe
        self.hands = [mp_hands.Hands(
            model_complexity=1, 
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5, 
            max_num_hands=2,
            static_image_mode=False
        ) for _ in range(2)]
        
        # Warm up models
        dummy = np.zeros((100, 100, 3), dtype=np.uint8)
        for hand in self.hands:
            try:
                hand.process(dummy)
            except Exception as e:
                logger.warning(f"MediaPipe warmup warning: {e}")

    def start(self):
        if self.running: return True
        
        logger.info(f"Starting controller with cameras {CAM_IDS}")
        
        if not self._open_cams(): 
            logger.error("Failed to open cameras")
            return False
        
        self.running = True
        threading.Thread(target=self._process_loop, daemon=True).start()
        
        logger.info("Controller started successfully")
        return True

    def start_control(self):
        if not self.running or self.control_active: return False
        
        # Check if the required arm is connected
        if not self.robot.is_arm_connected(self.arm_type):
            logger.warning(f"Cannot start control - {self.arm_type} arm not connected")
            return False
            
        logger.info("Starting gesture control...")
        success = self.robot.setup_control(self.arm_type)
        if success:
            self.control_active = True
            logger.info("Gesture control activated")
        return success

    def stop_control(self):
        if self.control_active:
            logger.info("Stopping gesture control...")
            self.control_active = False
            if self.robot.is_arm_connected(self.arm_type):
                self.robot.disable_torque(self.arm_type)
            logger.info("Gesture control deactivated")
        return True

    def pause_control(self):
        """Pause control without stopping it completely"""
        if self.control_active:
            logger.info("Pausing gesture control...")
            self.control_active = False
            logger.info("Gesture control paused")
        return True

    def _open_cams(self):
        """Open cameras with better error handling"""
        for i, cid in enumerate(CAM_IDS):
            try:
                cap = cv2.VideoCapture(cid)
                if not cap.isOpened():
                    logger.error(f"Camera {i+1} (ID: {cid}) failed to open")
                    # Try different backends
                    for backend in [cv2.CAP_V4L2, cv2.CAP_GSTREAMER]:
                        try:
                            cap = cv2.VideoCapture(cid, backend)
                            if cap.isOpened():
                                logger.info(f"Camera {i+1} opened with backend {backend}")
                                break
                        except:
                            continue
                    else:
                        return False
                
                # Set camera properties
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer for lower latency
                
                # Verify camera settings
                actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                logger.info(f"Camera {i+1}: {actual_width}x{actual_height}")
                
                self.cams[i] = cap
            except Exception as e:
                logger.error(f"Camera {i+1} error: {e}")
                return False
        return True

    def _predict(self, x, y, z):
        """Neural network prediction with safety checks"""
        if not self.params:
            # Return dummy positions that gradually move joints
            t = time.time()
            return np.array([
                2048 + int(500 * np.sin(t * 0.5)),
                2048 + int(300 * np.cos(t * 0.3)),
                2048 + int(200 * np.sin(t * 0.7)),
                2048 + int(100 * np.cos(t * 0.9))
            ])
        
        try:
            # Normalize and clamp inputs
            x_norm = max(0, min(1, x / 650.0))
            y_norm = max(0, min(1, y / 650.0))
            z_norm = max(0, min(1, z / 650.0))
            
            A = np.array([[x_norm], [y_norm], [z_norm]])
            L = len(self.params) // 2
            
            for l in range(1, L):
                A = np.maximum(0, self.params[f'W{l}'] @ A + self.params[f'b{l}'])
            
            output = ((self.params[f'W{L}'] @ A + self.params[f'b{L}']) * 4100).flatten()
            
            # Safety clamp outputs to reasonable joint limits
            output = np.clip(output, 1, 4094)
            
            return output.astype(int)
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return np.zeros(4, dtype=int)

    def _process_frame(self, idx):
        """Process frame with enhanced error handling"""
        if not self.cams[idx]:
            return self._create_dummy_frame(f"Camera {idx+1} not ready")
        
        # Try to read frame with retries
        frame = None
        for attempt in range(ATTEMPTS_READ_FRAME_COUNT):
            try:
                ret, f = self.cams[idx].read()
                if ret and f is not None:
                    frame = f
                    break
            except Exception as e:
                logger.warning(f"Camera {idx+1} read attempt {attempt+1} failed: {e}")
                time.sleep(0.01)
        
        if frame is None:
            return self._create_dummy_frame(f"Camera {idx+1}: No frame")

        try:
            h, w = frame.shape[:2]
            if h == 0 or w == 0:
                return self._create_dummy_frame(f"Camera {idx+1}: Invalid frame")
                
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process with MediaPipe
            results = self.hands[idx].process(rgb)
            
            self.hand_detected[idx] = False
            
            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                landmark = hand_landmarks.landmark[8]  # Index finger tip
                
                # Validate landmark coordinates
                if 0 <= landmark.x <= 1 and 0 <= landmark.y <= 1:
                    x, y = int(landmark.x * w), int(landmark.y * h)
                    self.tip[idx] = (x, y)
                    self.hand_detected[idx] = True
                    
                    if idx == 1:  # Z coordinate from second camera
                        self.z = max(0, min(y, self.height))  # Clamp Z value
                    
                    # Draw detection
                    cv2.circle(frame, (x, y), 10, (0, 0, 255), -1)
                               
            # Add status overlay
            status_color = (0, 255, 0) if self.hand_detected[idx] else (0, 0, 255)
            status_text = f"Cam{idx+1}: {'HAND' if self.hand_detected[idx] else 'NO HAND'}"
            cv2.putText(frame, status_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
                       
        except Exception as e:
            logger.error(f"Frame processing error camera {idx+1}: {e}")
            if frame is not None:
                cv2.putText(frame, f"Processing Error", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return frame

    def _create_dummy_frame(self, message):
        frame = np.zeros((self.height, self.width, 3), np.uint8)
        cv2.putText(frame, message, (70, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        return frame

    def _process_loop(self):
        while self.running:
            try:
                frames = [self._process_frame(i) for i in range(2)]
                positions = self.robot.get_positions(self.arm_type)
                
                # Emit status updates at controlled intervals
                current_time = time.time()
                if current_time - self.last_status_update >= self.status_update_interval:
                    self._emit_status_update()
                    self.last_status_update = current_time

                if self.control_active and (self.hand_detected[0] and self.hand_detected[1]) and self.robot.is_arm_connected(self.arm_type):
                    joints = self._predict(*self.tip[0], self.z)
                    if np.sum(np.abs(joints)) > 0:
                        self.robot.move(joints, self.arm_type)

                elif self.control_active and (not self.hand_detected[0] or not self.hand_detected[1]) and self.robot.is_arm_connected(self.arm_type):
                    
                    if not hasattr(self, 'last_hand_detected_time'):
                        self.last_hand_detected_time = current_time

                    if current_time - self.last_hand_detected_time >= HANDS_TIMEOUT:
                        self.robot.move(DEFAULT_JOINTS, self.arm_type)
                        self.last_hand_detected_time = current_time

                with self.data_lock:
                    self.last_frames = frames
                    self.last_data = [
                        self.tip[0][0] if self.hand_detected[0] else None, 
                        self.tip[0][1] if self.hand_detected[0] else None,
                        self.tip[1][0] if self.hand_detected[1] else None, 
                        self.tip[1][1] if self.hand_detected[1] else None
                    ] + positions
                    
                time.sleep(0.01)
            except Exception as e:
                logger.error(f"Processing error: {e}")
                time.sleep(0.1)
        self._cleanup()

    def _emit_status_update(self):
        data = self.get_last_data()
        safe_data = []
        for v in data:
            if v is None:
                safe_data.append(None)
            elif isinstance(v, np.generic):
                safe_data.append(v.item())
            else:
                safe_data.append(int(v) if isinstance(v, (int, float)) else v)
                
        headers = ["camera1_tip_x", "camera1_tip_y", "camera2_tip_x", "camera2_tip_y",
                 "follower_joint_1", "follower_joint_2", "follower_joint_3", "follower_joint_4"]
                 
        from app import socketio
        socketio.emit('status_update', dict(zip(headers, safe_data)))

    def _cleanup(self):
        self._cleanup_cameras()

    def _cleanup_cameras(self):
        for hand in self.hands:
            try: 
                hand.close()
            except: 
                pass
        for i, cam in enumerate(self.cams):
            if cam:
                try: 
                    cam.release()
                except: 
                    pass
        self.cams = [None, None]

    def get_last_frame(self, idx):
        with self.data_lock:
            frame = self.last_frames[idx]
            return frame.copy() if frame is not None else self._create_dummy_frame("No frame")

    def get_last_data(self):
        with self.data_lock:
            return self.last_data.copy()

    def stop(self):
        if not self.running: return
        logger.info("Stopping controller...")
        self.running = False
        self.control_active = False