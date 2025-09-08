#!/usr/bin/env python3
"""
Flask API routes module for REST endpoints.
"""

import os
import time
import numpy as np
import cv2
import logging
from datetime import datetime
from flask import Flask, render_template, Response, jsonify, request, send_file
from utils.csv_utils import get_csv_stats, save_snapshot_to_csv, CSV_FILENAME

logger = logging.getLogger(__name__)

def register_api_routes(app):
    """Register all API routes with the Flask app"""
    
    @app.route('/')
    def index():
        from config import ARDUINO_CONFIG, ROBOT_ARMS_CONFIG, CAMERA_CONFIG
        
        port_config = {
            'arduino_port': ARDUINO_CONFIG['port'],
            'follower_port': ROBOT_ARMS_CONFIG['follower']['port'],
            'leader_port': ROBOT_ARMS_CONFIG['leader']['port'],
            'camera1_id': CAMERA_CONFIG['camera1']['id'],
            'camera2_id': CAMERA_CONFIG['camera2']['id']
        }
        
        return render_template('index.html', **port_config)

    @app.route('/video_feed1')
    def video_feed1():
        return Response(generate_frames(0), mimetype='multipart/x-mixed-replace; boundary=frame')

    @app.route('/video_feed2')
    def video_feed2():
        return Response(generate_frames(1), mimetype='multipart/x-mixed-replace; boundary=frame')

    @app.route('/api/status')
    def api_status():
        """REST API endpoint for system status"""
        # Import here to avoid circular imports
        from app import system_initialized, controller, robot, arduino_controller
        
        if system_initialized and controller and controller.running:
            arduino_status = arduino_controller.get_status() if arduino_controller else {}
            return jsonify({
                'system_ready': True,
                'control_active': controller.control_active,
                'robot_connected': robot.is_connected,
                'connected_arms': list(robot.connected_arms),
                'hand_detected': controller.hand_detected,
                'data': controller.get_last_data(),
                'arduino': arduino_status
            })
        else:
            return jsonify({
                'system_ready': False,
                'control_active': False,
                'robot_connected': robot.is_connected if robot else False,
                'connected_arms': list(robot.connected_arms) if robot else [],
                'hand_detected': [False, False],
                'data': [None] * 8,
                'arduino': {}
            })

    @app.route('/api/csv/download')
    def api_download_csv():
        """REST API endpoint to download CSV file"""
        try:
            if os.path.exists(CSV_FILENAME):
                return send_file(CSV_FILENAME, 
                               as_attachment=True, 
                               download_name=f"robot_snapshots_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                               mimetype='text/csv')
            else:
                return jsonify({'error': 'CSV file not found'}), 404
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    @app.route('/api/csv/stats')
    def api_csv_stats():
        """REST API endpoint for CSV statistics"""
        try:
            stats = get_csv_stats()
            return jsonify(stats)
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    @app.route('/api/snapshot', methods=['POST'])
    def api_take_snapshot():
        """REST API endpoint to take snapshot"""
        try:
            # Import here to avoid circular imports
            from app import controller, robot
            
            if not controller or not controller.running:
                return jsonify({'success': False, 'error': 'System not ready'}), 400
            
            current_data = controller.get_last_data()
            if len(current_data) != 8:
                return jsonify({'success': False, 'error': 'Invalid data length'}), 400
            
            success, total_snapshots = save_snapshot_to_csv(current_data)
            
            if success:
                return jsonify({
                    'success': True,
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'data': current_data,
                    'total_snapshots': total_snapshots,
                    'connected_arms': list(robot.connected_arms) if robot else []
                })
            else:
                return jsonify({'success': False, 'error': 'Failed to save to CSV'}), 500
                
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 500

def generate_frames(cam_idx):
    """Generate frames for video streaming"""
    frame_count = 0
    while True:
        try:
            # Import here to avoid circular imports
            from app import controller, system_initialized
            
            if controller and controller.running and system_initialized:
                frame = controller.get_last_frame(cam_idx)
                if frame_count % 150 == 0:  # Log every 5 seconds at 30fps
                    logger.info(f"Camera {cam_idx + 1}: Frame delivered, shape: {frame.shape}")
            else:
                frame = np.zeros((480, 640, 3), np.uint8)
                status_msg = "System starting..."
                if not system_initialized:
                    status_msg = "System initializing..."
                elif not controller:
                    status_msg = "Controller not ready..."
                elif not controller.running:
                    status_msg = "Controller stopped..."
                    
                cv2.putText(frame, status_msg, (50, 240), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                
                if frame_count % 30 == 0:  # Log every second
                    logger.info(f"Camera {cam_idx + 1}: Status - {status_msg}")
                
            _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
            
            frame_count += 1
            if not controller or not controller.running:
                time.sleep(0.5)
            else:
                time.sleep(0.033)  # ~30 FPS
        except Exception as e:
            logger.error(f"Frame generation error for camera {cam_idx + 1}: {e}")
            time.sleep(0.5)