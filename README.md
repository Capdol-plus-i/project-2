# Robot Control System

A modularized robot control system with a web interface for real-time monitoring and control.

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install flask flask-socketio numpy opencv-python mediapipe pyaudio pyserial google-cloud-speech dynamixel-sdk
```

### 2. Configure Hardware
Edit the `config.py` file to match your hardware setup, including COM ports for the Arduino and robot arms, and camera IDs.

```python
# Example configuration in config.py
ARDUINO_CONFIG = {
    'port': 'COM4',
    # ...
}
ROBOT_ARMS_CONFIG = {
    'follower': {
        'port': 'COM5',
        # ...
    },
    'leader': {
        'port': 'COM3',
        # ...
    },
}
CAMERA_CONFIG = {
    'camera1': {
        'id': 2,
        # ...
    },
    'camera2': {
        'id': 3,
        # ...
    },
}
```

### 3. Start Application
```bash
python app.py
```

Visit `http://localhost:5000` in your browser to see the web interface.

## 📁 Project Structure

```
capdol-2/
├── app.py                          # Main application
├── config.py                       # All hardware and software configuration
├── controllers/
│   ├── arduino_controller.py       # Handles Arduino communication for LEDs and buttons
│   ├── voice_controller.py         # Handles speech recognition and voice commands
│   ├── robot_controller.py         # Manages camera feeds and hand gesture recognition
│   └── manipulator_robot.py        # Controls the Dynamixel servo-based robot arms
├── routes/
│   ├── api_routes.py               # Defines REST API endpoints
│   └── websocket_handlers.py       # Manages WebSocket events for real-time communication
├── utils/
│   └── csv_utils.py                # Utilities for saving and managing data in CSV files
├── templates/
│   └── index.html                  # Main HTML file for the web interface
├── src/
│   └── main.cpp                    # Arduino source code
├── platformio.ini                  # PlatformIO configuration for Arduino
└── README.md                       # This file
```

## ⚙️ Usage

### Web Interface
- **Real-time Video Feeds**: See live video from both cameras.
- **System Status**: Monitor the connection status of the robot, Arduino, and gesture control.
- **Control Buttons**: Start/stop gesture control and control the robot's mode.
- **Data Snapshots**: Manually save the robot's current state to a CSV file.

### Voice Commands (Korean)
- **"하이봇"**: Wake word to activate voice commands.
- **"시작"**: Start gesture control.
- **"정지"**: Stop gesture control.
- **"밝게" / "어둡게"**: Adjust the brightness of the LEDs connected to the Arduino.
- **"리셋"**: Reset the Arduino.
- **"종료"**: Stop the application.

### REST API
- `GET /api/status`: Get the current system status.
- `POST /api/snapshot`: Take a snapshot of the robot's current data.
- `GET /api/csv/download`: Download the collected data as a CSV file.
- `GET /api/csv/stats`: Get statistics about the CSV file (size, number of entries).

## 🔧 Troubleshooting

### Connection Issues
- **Check `config.py`**: Ensure that the COM ports and camera IDs in `config.py` are correct.
- **Hardware Connections**: Verify that the Arduino, robot arms, and cameras are properly connected to your computer.

### Import Errors
If you get an error about a missing module, make sure you have installed all the dependencies listed in the "Install Dependencies" section.

### Voice Recognition Issues
- **Google Cloud Credentials**: Ensure you have set up Google Cloud Speech-to-Text API credentials correctly in your environment.
- **Microphone**: Check that your microphone is working and accessible by the application.
