아래 내용을 그대로 `SETUP_NOTEBOOK.md`로 저장해 쓰면 딱 좋아요. (필요하면 파일명 바꿔도 무방)

---

# Jetson Orin Nano 로보틱스 프로젝트 세팅 종합 문서

* 플랫폼: **Jetson Orin Nano Developer Kit (Ubuntu + L4T)**
* 목적:

  * 2× USB 웹캠 → GStreamer H.264 RTP 송출 (Windows 수신/합성)
  * MediaPipe로 **손 랜드마크** 검출 및 **오버레이 영상** 송출
  * 손 좌표를 UDP(JSON)로 발행 → **Dynamixel 로봇팔** 추종
  * UNO(PlatformIO)로 **NeoPixel** 제어
  * Google Cloud Speech + VAD 기반 **음성 명령**
* 셸: zsh
* 리포지토리 베이스: [https://github.com/Capdol-plus-i/project-2](https://github.com/Capdol-plus-i/project-2)

---

## 목차

1. [필수 패키지 설치 (apt)](#필수-패키지-설치-apt)
2. [VS Code 원격 SSH 연결](#vs-code-원격-ssh-연결)
3. [Python 가상환경 구성](#python-가상환경-구성)
4. [오디오(마이크) 설정 & 테스트](#오디오마이크-설정--테스트)
5. [udev 규칙 (시리얼/카메라/마이크)](#udev-규칙-시리얼카메라마이크)
6. [카메라 프리셋(밝기/노출 등) 자동 적용](#카메라-프리셋밝기노출-등-자동-적용)
7. [GStreamer 송출 & Windows 수신/합성](#gstreamer-송출--windows-수신합성)
8. [MediaPipe 손 검출 + 오버레이 + 송출(gi/GStreamer)](#mediapipe-손-검출--오버레이--송출gigstreamer)
9. [통합 로거: 손 추적 + 로봇팔 동기화 + 데이터 수집](#통합-로거-손-추적--로봇팔-동기화--데이터-수집)
10. [Leader-Follower 로봇팔 동기화](#leader-follower-로봇팔-동기화)
11. [Arduino/NeoPixel 제어(PlatformIO)](#arduinoneopixel-제어platformio)
12. [음성 인식(웨이크워드+명령)](#음성-인식웨이크워드명령)
13. [부팅/실행 자동화(선택)](#부팅실행-자동화선택)
14. [자주 쓰는 확인/디버깅 명령](#자주-쓰는-확인디버깅-명령)

---

## 필수 패키지 설치 (apt)

```zsh
sudo apt update
sudo apt install -y \
  git curl unzip pkg-config build-essential cmake \
  python3-opencv python3-gi gir1.2-gst-1.0 \
  gstreamer1.0-tools gstreamer1.0-libav \
  gstreamer1.0-plugins-base gstreamer1.0-plugins-good \
  gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly \
  v4l-utils alsa-utils \
  libasound2-dev portaudio19-dev \
  udev
```

> **중요**: `python3-opencv`는 **GStreamer 지원 켜진 OpenCV**(시스템 빌드)입니다.

---

## VS Code 원격 SSH 연결

1. Jetson에 SSH 허용 (기본 열려 있음).
2. VS Code → 확장 “**Remote - SSH**” 설치
3. `F1` → **Remote-SSH: Connect to Host…** → `capdol@<JETSON_IP>`
4. 연결 후 좌측 Remote Explorer에서 워크스페이스 열기 (`~/project-2`)

---

## Python 가상환경 구성

### 통합 venv (**모든 기능 포함**)

```zsh
cd ~/project-2
python3 -m venv .venv_cv --system-site-packages
source .venv_cv/bin/activate

pip install --upgrade pip wheel setuptools
pip install "mediapipe==0.10.18" "protobuf<5,>=4.25.3" numpy==1.26.4 scipy==1.15.3
pip install pyaudio webrtcvad sounddevice dynamixel-sdk google-cloud-speech
pip install PyGObject pycairo  # GStreamer Python bindings
```

> 확인:
>
> ```zsh
> python - <<'PY'
> import cv2; print("GStreamer enabled? ->", "GStreamer" in cv2.getBuildInformation())
> import mediapipe; print("MediaPipe OK")
> import webrtcvad; print("webrtcvad OK")
> from dynamixel_sdk import *; print("Dynamixel SDK OK")
> from google.cloud import speech; print("Google Cloud Speech OK")
> import gi; gi.require_version('Gst', '1.0'); print("GStreamer Python bindings OK")
> PY
> ```
>
> 모두 OK면 통합 환경 완료.

### 프로젝트 구조

```
~/project-2/
├── unified_logger.py              # 통합 로거 (손 추적 + 로봇팔 + 데이터 수집)
├── leader_follower_sync.py        # Leader-Follower 로봇팔 동기화
├── voice_recognition_improved.py  # 향상된 음성 인식 시스템
├── hardware_config.json          # 하드웨어 설정
├── calibration.json              # 로봇팔 calibration 데이터
├── scripts/
│   ├── hand_overlay_stream.py    # 손 오버레이 스트리밍 (OpenCV)
│   ├── hand_overlay_stream_gst.py # 손 오버레이 스트리밍 (GStreamer)
│   ├── apply_cam_preset.sh       # 카메라 프리셋 적용
│   └── set_led.py               # LED 제어 스크립트
├── firmware/                     # Arduino 펌웨어
│   ├── platformio.ini
│   └── src/main.cpp
└── .venv_cv/                    # Python 가상환경
```

---

## 오디오(마이크) 설정 & 테스트

* 장치 확인:

  ```zsh
  arecord -l
  arecord -L
  ```
* 테스트(Blue Tiki `hw:4,0` 예시, 44.1kHz):

  ```zsh
  arecord -D plughw:4,0 -f cd -d 3 test.wav && aplay test.wav
  ```

**Python에서**는 입력 장치/샘플레이트를 장치에 맞춰 설정(Blue Tiki: 44100Hz 추천).

---

## udev 규칙 (시리얼/카메라/마이크)

### 1) Arduino(UNO) 시리얼 별칭 + 권한

UNO(Arduino) 예시: `VID=2341 PID=0043` → `/dev/arduino`

```zsh
sudo tee /etc/udev/rules.d/99-arduino.rules >/dev/null <<'EOF'
SUBSYSTEM=="tty", ATTRS{idVendor}=="2341", ATTRS{idProduct}=="0043", \
  SYMLINK+="arduino", MODE="0666", GROUP="dialout"
EOF

sudo udevadm control --reload-rules && sudo udevadm trigger
```

> 사용자 그룹에 `dialout` 포함 필요. `groups`로 확인.

### 2) Dynamixel USB-Serial (CH340) — **leader/follower** 고정

* **follower\_arm**: `VID=1a86 PID=55d3 serial=5970073211`
* **leader\_arm**:   `VID=1a86 PID=55d3 serial=5970073130`

```zsh
sudo tee /etc/udev/rules.d/99-dxl.rules >/dev/null <<'EOF'
# follower
SUBSYSTEM=="tty", ATTRS{idVendor}=="1a86", ATTRS{idProduct}=="55d3", \
  ATTRS{serial}=="5970073211", SYMLINK+="follower_arm", MODE="0666", GROUP="dialout"

# leader
SUBSYSTEM=="tty", ATTRS{idVendor}=="1a86", ATTRS{idProduct}=="55d3", \
  ATTRS{serial}=="5970073130", SYMLINK+="leader_arm", MODE="0666", GROUP="dialout"
EOF

sudo udevadm control --reload-rules && sudo udevadm trigger
```

### 3) 웹캠 좌/우 **by-path** 고정 심볼릭 링크

* 왼쪽(포트 **1-2.2**, index0) → `/dev/cam_left`
* 오른쪽(포트 **1-2.4**, index0) → `/dev/cam_right`

```zsh
sudo tee /etc/udev/rules.d/99-cams.rules >/dev/null <<'EOF'
# LEFT camera
KERNEL=="video[0-9]*", SUBSYSTEM=="video4linux", \
  ATTRS{idVendor}=="2ce3", ATTRS{idProduct}=="c670", \
  KERNELS=="1-2.2", ATTR{index}=="0", \
  SYMLINK+="cam_left", MODE="0666", GROUP="video"

# RIGHT camera
KERNEL=="video[0-9]*", SUBSYSTEM=="video4linux", \
  ATTRS{idVendor}=="2ce3", ATTRS{idProduct}=="c670", \
  KERNELS=="1-2.4", ATTR{index}=="0", \
  SYMLINK+="cam_right", MODE="0666", GROUP="video"
EOF

sudo udevadm control --reload-rules && sudo udevadm trigger
```

### 4) USB 마이크 별칭 (Blue Tiki)

```zsh
sudo tee /etc/udev/rules.d/99-mic.rules >/dev/null <<'EOF'
# Blue Tiki USB Mic (예: pcmC4D0c 노드 기준)
SUBSYSTEM=="sound", KERNEL=="pcmC4D0c", \
  ATTRS{idVendor}=="b58e", ATTRS{idProduct}=="8454", \
  KERNELS=="1-2.1", SYMLINK+="mic_main"
EOF

sudo udevadm control --reload-rules && sudo udevadm trigger
```

---

## 카메라 프리셋(밝기/노출 등) 자동 적용

**목표 값**

* `brightness=20, contrast=180, saturation=200, hue=0, gamma=80`
* `auto_exposure=Manual(1), exposure_time_absolute=78`

### 1) 즉시 적용 (양쪽)

```zsh
LEFT=/dev/v4l/by-path/platform-3610000.usb-usb-0:2.2:1.0-video-index0
RIGHT=/dev/v4l/by-path/platform-3610000.usb-usb-0:2.4:1.0-video-index0

sudo v4l2-ctl -d "$LEFT"  --set-ctrl=auto_exposure=1
sudo v4l2-ctl -d "$LEFT"  --set-ctrl=exposure_time_absolute=78
sudo v4l2-ctl -d "$LEFT"  --set-ctrl=brightness=20,contrast=180,saturation=200,hue=0,gamma=80

sudo v4l2-ctl -d "$RIGHT" --set-ctrl=auto_exposure=1
sudo v4l2-ctl -d "$RIGHT" --set-ctrl=exposure_time_absolute=78
sudo v4l2-ctl -d "$RIGHT" --set-ctrl=brightness=20,contrast=180,saturation=200,hue=0,gamma=80
```

> **순서 중요**: 수동노출 → 노출값 → 나머지.

### 2) 프로젝트 스크립트

`~/project-2/scripts/apply_cam_preset.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail
EXP=78; BRI=20; CON=180; SAT=200; HUE=0; GAM=80
DEVS=(
  "/dev/v4l/by-path/platform-3610000.usb-usb-0:2.2:1.0-video-index0"
  "/dev/v4l/by-path/platform-3610000.usb-usb-0:2.4:1.0-video-index0"
)
for DEV in "${DEVS[@]}"; do
  [[ -e "$DEV" ]] || { echo "[skip] $DEV"; continue; }
  echo "[cam-preset] $DEV"
  sleep 0.2
  v4l2-ctl -d "$DEV" --set-ctrl=auto_exposure=1 || true
  v4l2-ctl -d "$DEV" --set-ctrl=exposure_time_absolute=${EXP} || true
  v4l2-ctl -d "$DEV" --set-ctrl=brightness=${BRI},contrast=${CON},saturation=${SAT},hue=${HUE},gamma=${GAM} || true
  v4l2-ctl -d "$DEV" --get-ctrl=brightness,contrast,saturation,hue,gamma,auto_exposure,exposure_time_absolute || true
done
```

```zsh
chmod +x ~/project-2/scripts/apply_cam_preset.sh
# 사용
sudo ~/project-2/scripts/apply_cam_preset.sh
```

### 3) udev + systemd 자동화 (장치 연결 시마다)

`/usr/local/bin/set_cam_controls.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail
DEV="${1:?usage: set_cam_controls.sh /dev/videoX}"
EXP=78; BRI=20; CON=180; SAT=200; HUE=0; GAM=80
sleep 0.2
/usr/bin/v4l2-ctl -d "$DEV" --set-ctrl=auto_exposure=1 || true
/usr/bin/v4l2-ctl -d "$DEV" --set-ctrl=exposure_time_absolute=${EXP} || true
/usr/bin/v4l2-ctl -d "$DEV" --set-ctrl=brightness=${BRI},contrast=${CON},saturation=${SAT},hue=${HUE},gamma=${GAM} || true
/usr/bin/v4l2-ctl -d "$DEV" --get-ctrl=brightness,contrast,saturation,hue,gamma,auto_exposure,exposure_time_absolute || true
```

`/etc/systemd/system/cam-setup@.service`:

```ini
[Unit]
Description=Apply V4L2 preset to %I
After=dev-%i.device
Requires=dev-%i.device

[Service]
Type=oneshot
ExecStart=/usr/local/bin/set_cam_controls.sh /dev/%I
```

`/etc/udev/rules.d/99-cam-setup.rules`:

```udev
ACTION=="add", SUBSYSTEM=="video4linux", KERNEL=="video*", \
  ATTRS{idVendor}=="2ce3", ATTRS{idProduct}=="c670", KERNELS=="1-2.2", ATTR{index}=="0", \
  ENV{SYSTEMD_WANTS}="cam-setup@%k.service"

ACTION=="add", SUBSYSTEM=="video4linux", KERNEL=="video*", \
  ATTRS{idVendor}=="2ce3", ATTRS{idProduct}=="c670", KERNELS=="1-2.4", ATTR{index}=="0", \
  ENV{SYSTEMD_WANTS}="cam-setup@%k.service"
```

적용:

```zsh
sudo chmod +x /usr/local/bin/set_cam_controls.sh
sudo udevadm control --reload-rules
sudo systemctl daemon-reload
sudo udevadm trigger --subsystem-match=video4linux
```

---

## GStreamer 송출 & Windows 수신/합성

### 기본 카메라 스트리밍

#### Jetson → H.264 RTP (기본)

```zsh
export RX=<WINDOWS_IP>

# 왼쪽 카메라 (포트 5001, PT=96)
gst-launch-1.0 -e \
  v4l2src device=/dev/v4l/by-path/platform-3610000.usb-usb-0:2.2:1.0-video-index0 io-mode=2 ! \
  image/jpeg,width=1280,height=720,framerate=30/1 ! \
  jpegdec ! videoconvert ! queue leaky=downstream max-size-buffers=1 ! \
  x264enc tune=zerolatency speed-preset=ultrafast bitrate=4000 key-int-max=30 ! \
  h264parse config-interval=1 ! rtph264pay pt=96 mtu=1200 ! \
  udpsink host=$RX port=5001 sync=false async=false

# 오른쪽 카메라 (포트 5003, PT=97)
gst-launch-1.0 -e \
  v4l2src device=/dev/v4l/by-path/platform-3610000.usb-usb-0:2.4:1.0-video-index0 io-mode=2 ! \
  image/jpeg,width=1280,height=720,framerate=30/1 ! \
  jpegdec ! videoconvert ! queue leaky=downstream max-size-buffers=1 ! \
  x264enc tune=zerolatency speed-preset=ultrafast bitrate=4000 key-int-max=30 ! \
  h264parse config-interval=1 ! rtph264pay pt=97 mtu=1200 ! \
  udpsink host=$RX port=5003 sync=false async=false
```

### 고급 스트리밍 (스크립트 기반)

#### 1) OpenCV 기반 손 오버레이 스트리밍

`scripts/hand_overlay_stream.py` - MediaPipe 손 추적과 GStreamer 통합:

```zsh
source ~/project-2/.venv_cv/bin/activate

# 왼쪽 카메라 → 5001, 손 좌표를 UDP(5555)로 발행
python scripts/hand_overlay_stream.py \
  --rx 10.96.162.204 --port 5001 --pt 96 \
  --dev /dev/v4l/by-path/platform-3610000.usb-usb-0:2.2:1.0-video-index0 \
  --w 1280 --h 720 --fps 30 --bitrate 4000 --gop 30 \
  --send-xy 127.0.0.1:5555 --preview

# 오른쪽 카메라 → 5003
python scripts/hand_overlay_stream.py \
  --rx 10.96.162.204 --port 5003 --pt 97 \
  --dev /dev/v4l/by-path/platform-3610000.usb-usb-0:2.4:1.0-video-index0 \
  --w 1280 --h 720 --fps 30 --bitrate 4000 --gop 30
```

#### 2) GStreamer Python 기반 스트리밍

`scripts/hand_overlay_stream_gst.py` - gi/GObject로 파이프라인 직접 구성:

```zsh
# 왼쪽 카메라 (더 안정적인 GStreamer 파이프라인)
python scripts/hand_overlay_stream_gst.py \
  --rx 10.96.162.204 --port 5001 --pt 96 \
  --dev /dev/v4l/by-path/platform-3610000.usb-usb-0:2.2:1.0-video-index0 \
  --w 1280 --h 720 --fps 30 --bitrate 4000 --gop 30 \
  --send-xy 127.0.0.1:5555
```

> **권장**: OpenCV VideoWriter 이슈가 있을 때 `hand_overlay_stream_gst.py` 사용

### Windows 수신

#### 단일 카메라 수신

```powershell
# 왼쪽(5001, pt=96)
gst-launch-1.0 -v `
  udpsrc port=5001 caps="application/x-rtp, media=video, encoding-name=H264, payload=96, clock-rate=90000" ! `
  rtpjitterbuffer latency=60 ! rtph264depay ! h264parse ! d3d11h264dec ! d3d11videosink sync=false

# 오른쪽(5003, pt=97)
gst-launch-1.0 -v `
  udpsrc port=5003 caps="application/x-rtp, media=video, encoding-name=H264, payload=97, clock-rate=90000" ! `
  rtpjitterbuffer latency=60 ! rtph264depay ! h264parse ! d3d11h264dec ! d3d11videosink sync=false
```

#### 듀얼 카메라 합성 (2×1)

```powershell
gst-launch-1.0 -v `
  compositor name=mix background=black `
    sink_0::xpos=0   sink_0::ypos=0   sink_0::width=960  sink_0::height=540 `
    sink_1::xpos=960 sink_1::ypos=0   sink_1::width=960  sink_1::height=540 ! `
  videoconvert ! autovideosink sync=false `
  udpsrc port=5001 caps="application/x-rtp, media=video, encoding-name=H264, payload=96, clock-rate=90000" ! `
    rtpjitterbuffer latency=60 ! rtph264depay ! h264parse ! avdec_h264 ! queue ! videoconvert ! mix.sink_0 `
  udpsrc port=5003 caps="application/x-rtp, media=video, encoding-name=H264, payload=97, clock-rate=90000" ! `
    rtpjitterbuffer latency=60 ! rtph264depay ! h264parse ! avdec_h264 ! queue ! videoconvert ! mix.sink_1
```

> 하드웨어 디코딩 가능하면 `avdec_h264` 대신 `d3d11h264dec` 사용

---

## MediaPipe 손 검출 + 오버레이 + 송출

현재 프로젝트에는 두 가지 MediaPipe 손 검출 방식이 구현되어 있습니다:

### 1) 통합 로거 방식 (권장)

`unified_logger.py`는 모든 기능을 통합한 솔루션으로, 실시간 카메라 디스플레이와 데이터 수집을 제공합니다:

```zsh
source ~/project-2/.venv_cv/bin/activate

# 실시간 카메라 화면 + 손 추적 + 데이터 수집
python unified_logger.py --test  # 테스트 모드 (기록 안함)
python unified_logger.py --mode snapshot  # 스냅샷 모드
python unified_logger.py --mode continuous  # 연속 기록 모드

# 로봇팔 동기화 포함
python unified_logger.py --test --sync --cal-load calibration.json
```

**주요 특징:**
- **듀얼 카메라 지원**: 양손 동시 추적
- **실시간 오버레이**: 검지손가락 좌표와 랜드마크 표시
- **데이터 수집**: CSV 형태로 모든 데이터 저장
- **로봇팔 연동**: Leader-Follower 동기화 동시 실행

### 2) 스트리밍 전용 방식

네트워크 스트리밍이 필요한 경우 전용 스크립트 사용:

#### OpenCV 기반 스트리밍

`scripts/hand_overlay_stream.py` - OpenCV VideoWriter 사용:

```zsh
# 왼쪽 카메라 → Windows(5001 포트), 손 좌표를 UDP(5555)로 발행
python scripts/hand_overlay_stream.py \
  --rx 10.96.162.204 --port 5001 --pt 96 \
  --dev /dev/v4l/by-path/platform-3610000.usb-usb-0:2.2:1.0-video-index0 \
  --w 1280 --h 720 --fps 30 --bitrate 4000 --gop 30 \
  --send-xy 127.0.0.1:5555 --preview

# 오른쪽 카메라 → Windows(5003 포트)
python scripts/hand_overlay_stream.py \
  --rx 10.96.162.204 --port 5003 --pt 97 \
  --dev /dev/v4l/by-path/platform-3610000.usb-usb-0:2.4:1.0-video-index0 \
  --w 1280 --h 720 --fps 30 --bitrate 4000 --gop 30
```

#### GStreamer Python 기반 스트리밍

`scripts/hand_overlay_stream_gst.py` - gi/GObject로 파이프라인 직접 구성:

```zsh
# OpenCV VideoWriter 이슈 우회 (더 안정적)
python scripts/hand_overlay_stream_gst.py \
  --rx 10.96.162.204 --port 5001 --pt 96 \
  --dev /dev/v4l/by-path/platform-3610000.usb-usb-0:2.2:1.0-video-index0 \
  --w 1280 --h 720 --fps 30 --bitrate 4000 --gop 30 \
  --send-xy 127.0.0.1:5555
```

### MediaPipe 설정 최적화

**성능 조정:**
```python
# 높은 정확도 (높은 CPU 사용률)
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    model_complexity=1,  # 0-2 (높을수록 정확하지만 느림)
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# 성능 우선 (낮은 CPU 사용률)
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    model_complexity=0,  # 가장 빠름
    min_detection_confidence=0.5,
    min_tracking_confidence=0.3
)
```

**해상도 조정:**
```zsh
# 고해상도 (높은 품질, 높은 CPU)
export W=1280 H=720 FPS=30

# 최적화 (균형)
export W=960 H=540 FPS=24

# 성능 우선 (낮은 품질, 낮은 CPU)
export W=640 H=480 FPS=20
```

### 손 좌표 데이터 형식

#### UDP JSON 형식 (스트리밍용)
```json
{
  "timestamp": 1640995200.123,
  "hands": [
    {
      "landmarks": [[0.1, 0.2, 0.05], ...],  # 21개 랜드마크
      "index_tip": [120, 240],                # 검지 끝 픽셀 좌표
      "handedness": "Right"
    }
  ]
}
```

#### CSV 형식 (데이터 수집용)
```csv
timestamp,cam1_x,cam1_y,cam2_x,cam2_y,follower_pos1,follower_pos2,follower_pos3,follower_pos4
1640995200.123,120,240,135,255,2048,2100,1950,2048
```


---

## 통합 로거: 손 추적 + 로봇팔 동기화 + 데이터 수집

`unified_logger.py`는 모든 기능을 통합한 올인원 솔루션입니다:

### 주요 기능

1. **듀얼 카메라 손 추적**: MediaPipe로 양손 검지손가락 좌표 추출
2. **Leader-Follower 로봇팔 동기화**: 실시간 모터 위치 동기화
3. **실시간 카메라 디스플레이**: 손가락 좌표와 랜드마크 오버레이
4. **데이터 수집**: CSV 형태로 모든 데이터 통합 로깅
5. **Calibration 시스템**: 로봇팔 간 정확한 매핑

### 사용법

#### 기본 명령어

```zsh
source ~/project-2/.venv_cv/bin/activate

# 기본 연속 기록 (실시간 카메라 화면 포함)
python unified_logger.py --mode continuous

# 스냅샷 모드 (SPACE키로 수동 기록)
python unified_logger.py --mode snapshot

# 테스트 모드 (기록하지 않고 화면만)
python unified_logger.py --test

# 동기화 포함 모드
python unified_logger.py --sync --cal-load calibration.json

# 동기화 + 스냅샷 모드
python unified_logger.py --mode snapshot --sync --cal-load calibration.json
```

#### 명령행 옵션

- `--mode`: `continuous` (연속기록) 또는 `snapshot` (수동기록)
- `--sync`: Leader-Follower 동기화 활성화
- `--cal-load`: Calibration 파일 로드
- `--test`: 테스트 모드 (기록하지 않음)
- `--output`: 출력 CSV 파일명 지정

#### 실시간 화면 기능

- **검지손가락 끝**: 초록색 원과 좌표 텍스트로 표시
- **손 랜드마크**: MediaPipe 손 구조 전체 표시
- **상태 정보**: 화면 상단에 모드, 프레임 번호 등 표시
- **제어 안내**: 화면 하단에 키 조작 안내

#### 키 조작

- **연속 기록 모드**: `Ctrl+C` 또는 `ESC`로 종료
- **스냅샷 모드**: `SPACE`로 스냅샷, `ESC`로 종료
- **테스트 모드**: `Ctrl+C` 또는 `ESC`로 종료

#### 출력 데이터 형식

CSV 파일에 다음 컬럼으로 저장:
```
timestamp, cam1_x, cam1_y, cam2_x, cam2_y, follower_pos1, follower_pos2, follower_pos3, follower_pos4
```

### 하드웨어 설정

#### 카메라 설정
- Camera 1: `/dev/video0` (첫 번째 USB 카메라)
- Camera 2: `/dev/video2` (두 번째 USB 카메라, 선택사항)

#### 로봇팔 설정
- **Leader Arm**: `/dev/leader_arm` (XL330-M077-T x4)
- **Follower Arm**: `/dev/follower_arm` (XL430-W250-T x3 + XL330-M288-T x1)

설정은 `hardware_config.json`에서 수정 가능:

```json
{
  "robot_arms": {
    "leader": {"port": "/dev/leader_arm", "baudrate": 1000000},
    "follower": {"port": "/dev/follower_arm", "baudrate": 1000000}
  }
}
```

### Calibration 시스템

#### 자동 Calibration (권장)
```zsh
# leader_follower_sync.py 사용
python leader_follower_sync.py
# 프로그램 내에서: cal auto → cal save
```

#### 수동 Calibration
```zsh
# leader_follower_sync.py 사용
python leader_follower_sync.py
# 프로그램 내에서:
# cal zero (양쪽 팔을 센터로)
# cal 1 -5 (모터 1에 -5도 오프셋)
# cal save
```

#### Calibration 데이터 구조
```json
{
  "position_offsets": {"1": 0.0, "2": -5.2, "3": 3.1, "4": 0.0},
  "direction_multipliers": {"1": 1, "2": -1, "3": 1, "4": 1},
  "id_map": {"1": 1, "2": 2, "3": 3, "4": 4}
}
```

### 환경 변수

카메라 해상도/FPS 조정:
```zsh
export W=1280 H=720 FPS=30
python unified_logger.py --test
```

### 문제 해결

#### 카메라 문제
```zsh
# 카메라 장치 확인
v4l2-ctl --list-devices

# 카메라 포맷 확인
v4l2-ctl -d /dev/video0 --list-formats-ext

# 권한 확인
ls -l /dev/video*
```

#### 로봇팔 연결 문제
```zsh
# 시리얼 포트 확인
ls -l /dev/leader_arm /dev/follower_arm

# udev 규칙 재로드
sudo udevadm control --reload-rules
sudo udevadm trigger
```

#### MediaPipe 성능 최적화
- 해상도 낮추기: `W=640 H=480`
- FPS 낮추기: `FPS=20`
- 복잡도 낮추기: 코드에서 `model_complexity=0`

---

## Leader-Follower 로봇팔 동기화

독립적인 Leader-Follower 동기화 시스템으로, 실시간 로봇팔 추종 제어를 제공합니다.

### 사용법

```zsh
source ~/project-2/.venv_cv/bin/activate
python leader_follower_sync.py
```

### 주요 명령어

#### 동기화 제어
- `start`: 실시간 동기화 시작
- `stop`: 동기화 중지
- `status`: 현재 상태 확인

#### Calibration 명령어
```
cal auto        # 자동 calibration
cal zero        # 양쪽 팔을 센터 위치로 이동
cal 1 -5        # 모터 1에 -5도 오프셋 설정
cal reset       # 모든 오프셋 초기화
cal save        # Calibration 저장
cal load        # Calibration 로드
```

#### 매핑 명령어
```
map 1 2         # Leader 모터 1을 Follower 모터 2에 매핑
map reset       # 매핑을 기본(1:1)으로 초기화
```

#### 시스템 명령어
- `h`, `help`: 도움말 표시
- `c`, `clear`: 화면 지우기
- `q`, `quit`: 프로그램 종료

### 설정 파일

#### hardware_config.json
```json
{
  "robot_arms": {
    "leader": {"port": "/dev/leader_arm", "baudrate": 1000000},
    "follower": {"port": "/dev/follower_arm", "baudrate": 1000000}
  }
}
```

#### calibration.json (자동 생성)
```json
{
  "timestamp": 1640995200.0,
  "position_offsets": {"1": 0.0, "2": -5.2, "3": 3.1, "4": 0.0},
  "direction_multipliers": {"1": 1, "2": -1, "3": 1, "4": 1},
  "id_map": {"1": 1, "2": 2, "3": 3, "4": 4}
}
```

### 모터 구성

#### Leader Arm (XL330-M077-T x4)
- 모터 ID: 1, 2, 3, 4
- 중심값: 2048 (0도)
- 해상도: 0.088도/unit

#### Follower Arm (XL430-W250-T x3 + XL330-M288-T x1)
- 모터 1-3: XL430-W250-T
- 모터 4: XL330-M288-T
- 중심값: 2048 (0도)
- 해상도: 0.088도/unit

### 안전 기능

- **에러 처리**: 연속 에러 시 자동 동기화 중지
- **토크 관리**: Follower 모터 토크 자동 on/off
- **신호 처리**: Ctrl+C로 안전한 종료
- **범위 제한**: 모터 위치 0-4095 범위 제한

---

## Arduino/NeoPixel 제어(PlatformIO)

* 의존성:

  * `TaskScheduler` (nicohood/PinChangeInterrupt는 1.2.9 권장)
  * `Adafruit NeoPixel`

```zsh
# 라이브러리 설치 예시
pio pkg install --library "TaskScheduler"
pio pkg install --library "Adafruit NeoPixel"
```

* 업로드 포트 고정: `/dev/arduino`(udev 별칭).
  `platformio.ini`에:

  ```ini
  upload_port = /dev/arduino
  monitor_port = /dev/arduino
  ```

* Jetson → UNO로 색 전송 예시:

  ```zsh
  python scripts/set_led.py 0 255 80
  ```

---

## 음성 인식(웨이크워드+명령)

### 향상된 음성 인식 시스템

`voice_recognition_improved.py`는 개선된 음성 인식 시스템을 제공합니다:

#### 주요 특징

- **듀얼 모드**: 웨이크워드 대기 → 명령 인식
- **로컬 VAD**: WebRTC VAD로 무음 구간 필터링
- **컨텍스트 힌트**: 웨이크워드/명령어 힌트로 인식률 향상
- **통합 제어**: NeoPixel LED + 로봇팔 토크 제어
- **실시간 피드백**: 상태별 LED 색상 변경

#### 사용법

```zsh
source ~/project-2/.venv_cv/bin/activate
python voice_recognition_improved.py
```

### Google Cloud Speech 인증 설정

1. **Google Cloud Console에서 서비스 계정 키 생성:**
   - Google Cloud Console → IAM & Admin → Service Accounts
   - Speech-to-Text API 권한이 있는 서비스 계정 생성/선택
   - 키 생성 → JSON 다운로드

2. **키 파일을 Jetson에 설정:**

```zsh
# 키 파일을 안전한 위치에 저장
mkdir -p ~/.config/gcloud
# 다운로드한 service-account-key.json을 ~/.config/gcloud/에 복사

# 환경변수 설정 (영구 적용)
echo 'export GOOGLE_APPLICATION_CREDENTIALS="$HOME/.config/gcloud/service-account-key.json"' >> ~/.zshrc
source ~/.zshrc

# 테스트
python -c "from google.cloud import speech; print('✓ 인증 성공')"
```

3. **API 활성화 확인:**
   - Google Cloud Console에서 Speech-to-Text API 활성화 확인

### 음성 명령어 체계

#### 웨이크워드
- "헤이 로봇", "로봇아", "시작해"

#### 제어 명령어

**로봇팔 토크 제어:**
- "토크 켜" / "토크 온" → 모든 모터 토크 활성화
- "토크 꺼" / "토크 오프" → 모든 모터 토크 비활성화

**LED 색상 제어:**
- "빨간불" → 빨간색 (255, 0, 0)
- "파란불" → 파란색 (0, 0, 255)
- "초록불" → 초록색 (0, 255, 0)
- "노란불" → 노란색 (255, 255, 0)
- "하얀불" → 흰색 (255, 255, 255)
- "불 꺼" → LED 끄기 (0, 0, 0)

### 시스템 설정

#### 마이크 설정
```zsh
# Blue Tiki USB 마이크 (44.1kHz 권장)
# voice_recognition_improved.py에서 자동으로 샘플레이트 설정
```

#### 하드웨어 연결 확인
```zsh
# Arduino 연결 확인
ls -l /dev/arduino

# 로봇팔 연결 확인
ls -l /dev/leader_arm /dev/follower_arm

# 마이크 확인
arecord -l
```

### 음성 처리 파라미터

#### VAD 설정
```python
# voice_recognition_improved.py 내부 설정
VAD_MODE = 3  # 0-3 (3이 가장 aggressive)
SAMPLE_RATE = 44100  # Blue Tiki 마이크 기준
CHUNK_DURATION_MS = 30  # VAD 청크 크기
```

#### Google Cloud Speech 설정
```python
# 언어 및 모델 설정
config = speech.RecognitionConfig(
    encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
    sample_rate_hertz=44100,
    language_code="ko-KR",
    model="command_and_search",  # 명령어 인식에 최적화
    speech_contexts=[speech.SpeechContext(phrases=wake_words + commands)]
)
```

### 상태 피드백

#### LED 상태 표시
- **파란색**: 웨이크워드 대기 중
- **노란색**: 명령어 인식 중
- **초록색**: 명령 실행 성공
- **빨간색**: 오류 발생
- **보라색**: 시스템 초기화 중

#### 터미널 출력
```
🎤 음성 인식 시스템 시작...
💙 웨이크워드 대기 중... (말해보세요: "헤이 로봇")
🎯 웨이크워드 감지: "헤이 로봇"
💛 명령어를 말씀하세요...
✅ 명령 실행: "토크 켜" → 모든 모터 토크 활성화
💙 웨이크워드 대기 중...
```

### 문제 해결

#### 마이크 문제
```zsh
# 마이크 장치 확인
arecord -l

# 테스트 녹음
arecord -D plughw:4,0 -f cd -d 3 test.wav && aplay test.wav
```

#### 인증 문제
```zsh
# 환경변수 확인
echo $GOOGLE_APPLICATION_CREDENTIALS

# 키 파일 권한 확인
ls -l ~/.config/gcloud/service-account-key.json
```

#### 하드웨어 연결 문제
```zsh
# udev 규칙 재로드
sudo udevadm control --reload-rules
sudo udevadm trigger

# 시리얼 포트 권한 확인
groups  # dialout 그룹 포함 확인
```

---

## 부팅/실행 자동화(선택)

### 자동화된 구성 요소

#### 이미 자동화된 항목
- **카메라 프리셋**: udev+systemd로 자동 적용
- **하드웨어 별칭**: udev 규칙으로 고정 심볼릭 링크

#### 수동 실행 권장 항목

**통합 로거 시스템:**
```zsh
# 데이터 수집 + 실시간 화면
python unified_logger.py --mode continuous --sync --cal-load calibration.json

# 테스트/모니터링
python unified_logger.py --test --sync
```

**독립 시스템들:**
```zsh
# 로봇팔 동기화만
python leader_follower_sync.py

# 음성 인식만
python voice_recognition_improved.py

# 네트워크 스트리밍만
python scripts/hand_overlay_stream.py --rx <WINDOWS_IP> --port 5001 --pt 96
```

### Systemd 서비스 구성 (선택적)

스트리밍/로봇 제어를 자동화하려면 systemd 서비스로 구성 가능:

#### 1) 통합 로거 서비스

`/etc/systemd/system/unified-logger.service`:
```ini
[Unit]
Description=Unified Hand Tracking and Robot Arm Logger
After=multi-user.target
Wants=multi-user.target

[Service]
Type=simple
User=capdol
Group=capdol
WorkingDirectory=/home/capdol/project-2
Environment=PATH=/home/capdol/project-2/.venv_cv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
ExecStart=/home/capdol/project-2/.venv_cv/bin/python unified_logger.py --mode continuous --sync --cal-load calibration.json
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

#### 2) 음성 인식 서비스

`/etc/systemd/system/voice-recognition.service`:
```ini
[Unit]
Description=Voice Recognition System
After=multi-user.target
Wants=multi-user.target

[Service]
Type=simple
User=capdol
Group=capdol
WorkingDirectory=/home/capdol/project-2
Environment=PATH=/home/capdol/project-2/.venv_cv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
Environment=GOOGLE_APPLICATION_CREDENTIALS=/home/capdol/.config/gcloud/service-account-key.json
ExecStart=/home/capdol/project-2/.venv_cv/bin/python voice_recognition_improved.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

#### 3) 스트리밍 서비스 (듀얼 카메라)

`/etc/systemd/system/hand-stream@.service`:
```ini
[Unit]
Description=Hand Overlay Streaming for Camera %i
After=multi-user.target
Wants=multi-user.target

[Service]
Type=simple
User=capdol
Group=capdol
WorkingDirectory=/home/capdol/project-2
Environment=PATH=/home/capdol/project-2/.venv_cv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
ExecStart=/home/capdol/project-2/.venv_cv/bin/python scripts/hand_overlay_stream.py --rx 10.96.162.204 --port %i --pt 96 --dev /dev/v4l/by-path/platform-3610000.usb-usb-0:2.2:1.0-video-index0 --w 1280 --h 720 --fps 30 --bitrate 4000 --gop 30
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

#### 서비스 관리

```zsh
# 서비스 활성화
sudo systemctl enable unified-logger.service
sudo systemctl enable voice-recognition.service

# 서비스 시작/중지
sudo systemctl start unified-logger.service
sudo systemctl stop unified-logger.service

# 서비스 상태 확인
sudo systemctl status unified-logger.service

# 로그 확인
sudo journalctl -u unified-logger.service -f
```

### 권장 운영 방식

#### 개발/테스트 단계
- 수동 실행으로 각 기능 테스트
- 터미널에서 직접 실행하여 디버깅

#### 운영 단계
- 안정화된 기능만 systemd 자동화
- 로그 모니터링 시스템 구축
- 자동 재시작 정책 적용

#### 하이브리드 방식 (권장)
```zsh
# 기본 시스템 자동 시작
sudo systemctl enable voice-recognition.service

# 데이터 수집은 필요시 수동 실행
python unified_logger.py --mode snapshot --sync
```

---

## 자주 쓰는 확인/디버깅 명령

### 하드웨어 상태 확인

#### 카메라
```zsh
# 카메라 장치 나열
v4l2-ctl --list-devices
ls -l /dev/video* /dev/v4l/by-path

# 카메라 심볼릭 링크 확인
ls -l /dev/cam_left /dev/cam_right

# 카메라 포맷 및 해상도 확인
v4l2-ctl --device=/dev/video0 --list-formats-ext

# 현재 카메라 설정 확인
v4l2-ctl -d /dev/video0 --all
v4l2-ctl -d /dev/video0 --get-ctrl=brightness,contrast,saturation,auto_exposure,exposure_time_absolute
```

#### 로봇팔 시리얼 포트
```zsh
# 로봇팔 포트 확인
ls -l /dev/leader_arm /dev/follower_arm
ls -l /dev/ttyUSB*

# 시리얼 장치 정보
udevadm info -a -n /dev/leader_arm
udevadm info -a -n /dev/follower_arm

# udev 규칙 테스트
sudo udevadm test /sys/class/tty/ttyUSB0
```

#### 오디오/마이크
```zsh
# ALSA 장치 확인
arecord -l
arecord -L

# 마이크 테스트
arecord -D plughw:4,0 -f cd -d 3 test.wav && aplay test.wav

# 마이크 심볼릭 링크 확인
ls -l /dev/mic_main
```

#### Arduino
```zsh
# Arduino 포트 확인
ls -l /dev/arduino
ls -l /dev/ttyACM*

# PlatformIO 장치 확인
pio device list
```

### 소프트웨어 상태 확인

#### Python 환경
```zsh
# 가상환경 활성화 확인
which python
echo $VIRTUAL_ENV

# 주요 패키지 버전 확인
python - <<'PY'
import cv2; print("OpenCV:", cv2.__version__)
import mediapipe; print("MediaPipe:", mediapipe.__version__)
import dynamixel_sdk; print("Dynamixel SDK: OK")
from google.cloud import speech; print("Google Cloud Speech: OK")
print("GStreamer in OpenCV:", "GStreamer" in cv2.getBuildInformation())
PY
```

#### GStreamer
```zsh
# GStreamer 플러그인 확인
gst-inspect-1.0 x264enc | head
gst-inspect-1.0 v4l2src | head
gst-inspect-1.0 rtph264pay | head

# GStreamer 버전
gst-launch-1.0 --version

# Python GStreamer 바인딩 확인
python - <<'PY'
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst
print("GStreamer Python bindings: OK")
PY
```

### 프로세스 및 네트워크 확인

#### 실행 중인 프로세스
```zsh
# Python 스크립트 확인
ps aux | grep python

# 특정 스크립트 확인
ps aux | grep unified_logger
ps aux | grep voice_recognition
ps aux | grep hand_overlay

# 포트 사용 확인
sudo netstat -tulpn | grep :5001
sudo netstat -tulpn | grep :5555
```

#### systemd 서비스
```zsh
# 서비스 상태 확인
sudo systemctl status unified-logger.service
sudo systemctl status voice-recognition.service

# 서비스 로그 확인
sudo journalctl -u unified-logger.service -f
sudo journalctl -u voice-recognition.service --since "1 hour ago"
```

### 빠른 기능 테스트

#### 카메라 기본 테스트
```zsh
# 기본 카메라 스트림 (5초)
gst-launch-1.0 v4l2src device=/dev/video0 ! autovideosink

# MJPG 포맷 테스트
gst-launch-1.0 v4l2src device=/dev/video0 ! image/jpeg,width=1280,height=720 ! jpegdec ! autovideosink
```

#### 로봇팔 연결 테스트
```zsh
# 간단한 연결 테스트
python - <<'PY'
from dynamixel_sdk import *
port = PortHandler("/dev/leader_arm")
packet = PacketHandler(2.0)
if port.openPort():
    print("Leader arm: Connected")
    port.closePort()
else:
    print("Leader arm: Failed")

port = PortHandler("/dev/follower_arm")
if port.openPort():
    print("Follower arm: Connected")
    port.closePort()
else:
    print("Follower arm: Failed")
PY
```

#### 음성 인식 테스트
```zsh
# Google Cloud 인증 확인
python -c "from google.cloud import speech; print('Google Cloud Speech: OK')"

# 마이크 입력 테스트
python - <<'PY'
import pyaudio
p = pyaudio.PyAudio()
print("Audio devices:")
for i in range(p.get_device_count()):
    info = p.get_device_info_by_index(i)
    print(f"  {i}: {info['name']} (inputs: {info['maxInputChannels']})")
p.terminate()
PY
```

### 문제 해결 명령

#### 권한 문제
```zsh
# 사용자 그룹 확인
groups

# dialout 그룹 추가 (필요시)
sudo usermod -a -G dialout $USER

# video 그룹 추가 (필요시)
sudo usermod -a -G video $USER
```

#### udev 규칙 재로드
```zsh
# udev 규칙 재로드
sudo udevadm control --reload-rules
sudo udevadm trigger

# 특정 서브시스템만 트리거
sudo udevadm trigger --subsystem-match=tty
sudo udevadm trigger --subsystem-match=video4linux
```

#### 로그 수집
```zsh
# 시스템 로그
sudo journalctl --since "1 hour ago" | grep -E "(video|tty|usb)"

# dmesg (하드웨어 관련)
dmesg | tail -50

# 특정 장치 로그
dmesg | grep -i usb
dmesg | grep -i video
```

---

### 참고/주의사항

#### Python 환경
- OpenCV **시스템 빌드**를 쓰기 위해 venv는 `--system-site-packages`로 생성
- MediaPipe는 CPU 사용량이 높음. 시작 시 해상도 낮추고 복잡도 0으로 설정 권장

#### 성능 최적화
- **해상도**: 시작은 `--w 960 --h 540 --fps 20` 권장
- **MediaPipe**: `model_complexity=0`, `max_num_hands=1` 사용
- **GStreamer**: `bitrate=2000-4000`, `tune=zerolatency` 설정

#### 네트워크 스트리밍
- Windows 수신 지연은 `rtpjitterbuffer latency=40~60`에서 조정
- 패킷 손실 시 `mtu=1200` 또는 더 낮게 설정
- 방화벽에서 UDP 포트 5001, 5003, 5555 허용

#### 로봇팔 안전
- **토크 상태**: 반드시 토크 ON 여부 확인 후 목표각 전송
- **각도 제한**: 안전을 위해 각도/속도 제한을 보수적으로 설정
- **에러 처리**: 연속 통신 에러 시 자동으로 토크 OFF

#### 하드웨어 주의사항
- **카메라 설정**: 파이프라인 포맷 전환 시 컨트롤 초기화 가능하므로 스트리밍 직전 재적용
- **USB 전력**: 다수 장치 연결 시 USB 허브 전력 부족 주의
- **시리얼 충돌**: 동시에 여러 프로그램이 같은 시리얼 포트 접근 시 충돌 발생

#### 데이터 수집
- **저장 공간**: 연속 기록 시 디스크 용량 모니터링 필요
- **백업**: 중요한 데이터는 정기적으로 백업
- **파일명**: 타임스탬프 기반 자동 파일명으로 덮어쓰기 방지

#### 음성 인식
- **API 비용**: Google Cloud Speech API 사용량 모니터링
- **네트워크**: 인터넷 연결 필요 (로컬 VAD는 오프라인 동작)
- **마이크 위치**: 주변 소음 최소화를 위한 마이크 위치 조정

#### 시스템 안정성
- **메모리**: 장시간 실행 시 메모리 사용량 모니터링
- **온도**: 고부하 작업 시 Jetson 온도 확인
- **전력**: 모든 장치 동시 사용 시 전력 공급 충분한지 확인

---

### 업데이트 로그

**2024년 최신 업데이트:**
- `unified_logger.py`: 통합 로깅 시스템 추가
- `leader_follower_sync.py`: 로봇팔 동기화 시스템 완성
- `voice_recognition_improved.py`: 향상된 음성 인식 시스템
- 실시간 카메라 디스플레이 기능 추가
- 스냅샷 모드 및 연속 기록 모드 구현
- Calibration 시스템 자동화
- systemd 서비스 설정 가이드 추가
- 종합적인 디버깅 명령어 모음 추가

---