# Integrated Hand Tracking & Robot Arm Logger

This system simultaneously records hand landmark coordinates from dual cameras and robot arm positions, creating synchronized datasets for machine learning and analysis.

## Data Format

The system records 8 data points per sample:
- `cam1_x`, `cam1_y`: Index finger tip coordinates from camera 1
- `cam2_x`, `cam2_y`: Index finger tip coordinates from camera 2
- `follower_pos1-4`: Position values from follower robot arm motors 1-4

## Quick Start

```bash
# RECOMMENDED: Complete system launcher (handles port conflicts)
python3 start_logging_system.py

# Manual startup (3 terminals required):
# Terminal 1: Robot data server
python3 shared_robot_interface.py

# Terminal 2: Robot synchronization
python3 leader_follower_sync.py
# Type 'start' to begin sync

# Terminal 3: Data logger
python3 run_logger.py
```

## Prerequisites

### Hardware Setup
1. **Dual Cameras**: Two USB cameras connected
   - Camera 1: `/dev/video0`
   - Camera 2: `/dev/video2`

2. **Robot Arms**: Leader and follower arms connected
   - Leader arm: Connected and readable
   - Follower arm: Connected and synchronized

### Software Prerequisites
```bash
# Required Python packages
pip install opencv-python mediapipe dynamixel-sdk numpy

# Make sure leader_follower_sync.py is running
python3 leader_follower_sync.py
# In another terminal, start sync with 'start' command
```

## Recording Modes

### 1. Continuous Recording Mode
```bash
python3 integrated_logger.py --mode continuous
```
- Records data continuously at ~10Hz
- Press Ctrl+C to stop
- Good for collecting large datasets

### 2. Snapshot Mode
```bash
python3 integrated_logger.py --mode snapshot
```
- Records data only when you press SPACE
- Press ESC to quit
- Good for collecting specific poses/positions

## Output

Data is saved as CSV files with timestamps:
```
timestamp,cam1_x,cam1_y,cam2_x,cam2_y,follower_pos1,follower_pos2,follower_pos3,follower_pos4
1634567890.123,320,240,315,245,2048,2100,1950,2000
```

## Troubleshooting

### Camera Issues
```bash
# Test cameras only
python3 hand_landmark_demo.py --headless

# Check available cameras
ls /dev/video*
```

### Robot Arm Issues
```bash
# Check robot arm communication
python3 leader_follower_sync.py

# Check available ports
ls /dev/ttyACM* /dev/leader_arm
```

### Common Problems

1. **"Camera initialization failed"**
   - Check camera connections
   - Try different camera indices (0, 1, 2, 3)

2. **"Robot arm initialization failed"**
   - Make sure leader_follower_sync.py is running
   - Check robot arm power and connections
   - Verify port names in hardware_config.json

3. **"MediaPipe initialization failed"**
   - Install: `pip install mediapipe`

4. **"No hand detected"**
   - Ensure good lighting
   - Position hands clearly in camera view
   - Check MediaPipe confidence thresholds

## File Structure

- `integrated_logger.py`: Main logging system
- `run_logger.py`: Easy launcher script
- `hand_landmark_demo.py`: Camera/hand tracking component
- `leader_follower_sync.py`: Robot arm synchronization
- `hardware_config.json`: Hardware configuration

## Tips

1. **For best results:**
   - Good lighting for hand tracking
   - Robot arms properly calibrated
   - Stable camera mounting

2. **Data quality:**
   - Test individual components first
   - Check data output before long recordings
   - Use snapshot mode for verification

3. **Performance:**
   - Close unnecessary applications
   - Use SSD for fast data writing
   - Monitor CPU usage during recording

## Example Workflow

### Option 1: Easy Setup (Recommended)
```bash
# 1. Start complete system
python3 start_logging_system.py
# Choose option 1: "Start complete system"

# 2. Follow on-screen instructions:
#    - Start leader_follower_sync.py in another terminal
#    - Type 'start' to begin synchronization
#    - Press ENTER to continue

# 3. Choose logging mode and start collecting data
```

### Option 2: Manual Setup
```bash
# Terminal 1: Robot data server (avoids port conflicts)
python3 shared_robot_interface.py

# Terminal 2: Robot arm synchronization
python3 leader_follower_sync.py
# Type 'start' to begin sync

# Terminal 3: Data logger
python3 run_logger.py
# Choose continuous or snapshot mode
```

## Troubleshooting Port Conflicts

**Error: "device reports readiness to read but returned no data"**
- This happens when multiple programs try to access the same robot port
- **Solution**: Use the new 3-terminal approach above
- The `shared_robot_interface.py` acts as a single point of robot access
- Multiple clients can safely connect to it without conflicts