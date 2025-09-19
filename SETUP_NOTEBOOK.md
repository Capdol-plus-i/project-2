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
9. [손 좌표 → Dynamixel 로봇팔 추종 노드](#손-좌표--dynamixel-로봇팔-추종-노드)
10. [Arduino/NeoPixel 제어(PlatformIO)](#arduinoneopixel-제어platformio)
11. [음성 인식(웨이크워드+명령)](#음성-인식웨이크워드명령)
12. [부팅/실행 자동화(선택)](#부팅실행-자동화선택)
13. [자주 쓰는 확인/디버깅 명령](#자주-쓰는-확인디버깅-명령)

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
> PY
> ```
>
> 모두 OK면 통합 환경 완료.

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

### Jetson → H.264 RTP

(예: **왼쪽** 카메라 1280×720\@30, 포트 **5001**, PT=96)

```zsh
export RX=<WINDOWS_IP>

gst-launch-1.0 -e \
  v4l2src device=/dev/v4l/by-path/platform-3610000.usb-usb-0:2.2:1.0-video-index0 io-mode=2 ! \
  image/jpeg,width=1280,height=720,framerate=30/1 ! \
  jpegdec ! videoconvert ! queue leaky=downstream max-size-buffers=1 ! \
  x264enc tune=zerolatency speed-preset=ultrafast bitrate=4000 key-int-max=30 ! \
  h264parse config-interval=1 ! rtph264pay pt=96 mtu=1200 ! \
  udpsink host=$RX port=5001 sync=false async=false
```

**오른쪽** 카메라(포트 **5003**, PT=97)도 동일하게.

### Windows 수신(각각)

```powershell
# 왼쪽(5001, pt=96)
gst-launch-1.0 -v `
  udpsrc port=5001 caps="application/x-rtp, media=video, encoding-name=H264, payload=96, clock-rate=90000" ! `
  rtpjitterbuffer latency=60 ! rtph264depay ! h264parse ! d3d11h264dec ! d3d11videosink sync=false
```

### Windows 두 화면 합성(2×1)

```powershell
gst-launch-1.0 -v `
  compositor name=mix background=black `
    sink_0::xpos=0   sink_0::ypos=0   sink_0::width=960  sink_0::height=540 ! `
  videoconvert ! autovideosink sync=false `
  udpsrc port=5001 caps="application/x-rtp, media=video, encoding-name=H264, payload=96,  clock-rate=90000" ! `
    rtpjitterbuffer latency=60 ! rtph264depay ! h264parse ! avdec_h264 ! queue ! videoconvert ! mix.sink_0 `
  udpsrc port=5003 caps="application/x-rtp, media=video, encoding-name=H264, payload=97,  clock-rate=90000" ! `
    rtpjitterbuffer latency=60 ! rtph264depay ! h264parse ! avdec_h264 ! queue ! videoconvert ! mix.sink_1
```

> 하드웨어 디코딩 가능하면 `avdec_h264` 대신 `d3d11h264dec` 사용.

---

## MediaPipe 손 검출 + 오버레이 + 송출(gi/GStreamer)

OpenCV `VideoWriter(CAP_GSTREAMER)` 이슈를 우회하여 **gi(GObject)로 GStreamer 파이프라인 직접 구성**.

`scripts/hand_overlay_stream_gst.py`

```python
#!/usr/bin/env python3
import os; os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import gi; gi.require_version('Gst', '1.0')
from gi.repository import Gst, GObject
import sys, time, argparse, json, socket
import cv2, numpy as np

Gst.init(None)

# (인자 파서 생략) — 본문은 대화 내용의 최신 버전 사용
# 핵심 아이디어:
# - 입력: OpenCV VideoCapture (GStreamer → 실패 시 V4L2)
# - MediaPipe로 손 랜드마크 오버레이
# - 출력: gi 기반 GStreamer 파이프라인(appsrc→x264enc→rtph264pay→udpsink)
# - 옵션: --send-xy HOST:PORT 로 손 좌표(JSON) UDP 송신

# 전체 스크립트는 대화에서 제공한 최신본 사용
```

실행 예:

```zsh
source ~/project-2/.venv_cv/bin/activate

# 왼쪽 카메라 → 5001, 좌표를 로컬 UDP(5555)로 발행
python scripts/hand_overlay_stream_gst.py \
  --rx 10.96.162.204 --port 5001 --pt 96 \
  --dev /dev/v4l/by-path/platform-3610000.usb-usb-0:2.2:1.0-video-index0 \
  --w 1280 --h 720 --fps 30 --bitrate 4000 --gop 30 \
  --send-xy 127.0.0.1:5555 --preview

# 오른쪽 카메라 → 5003
python scripts/hand_overlay_stream_gst.py \
  --rx 10.96.162.204 --port 5003 --pt 97 \
  --dev /dev/v4l/by-path/platform-3610000.usb-usb-0:2.4:1.0-video-index0 \
  --w 1280 --h 720 --fps 30 --bitrate 4000 --gop 30
```


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

### 음성 처리 설정

* VAD: `webrtcvad==2.0.10`
* 마이크 샘플레이트: **장치 기본(예: 44100Hz)**에 맞추어 코드 상수 변경
* Google Cloud Speech:
  * 모델: `"command_and_search"`, 언어: `"ko-KR"`
  * speech context에 웨이크워드/명령 집합
* 기본 로직:
  1. VAD로 무음 제거하며 웨이크워드 스트리밍
  2. 웨이크워드 감지 → 짧은 **명령** 스트리밍 → 매칭 → 제어(토크 on/off, LED 색 등)

---

## 부팅/실행 자동화(선택)

* 카메라 프리셋: 위 **udev+systemd** 자동화로 이미 처리
* 스트리밍/손 좌표/팔 추종 실행 자동화는 `systemd` 서비스 2\~3개로 나누는 걸 권장

  * `vision@left.service`, `vision@right.service`
  * `arm_follower.service`
  * `voice_recognition.service`
  * 모두 통합 venv(`.venv_cv`) 사용 가능

---

## 자주 쓰는 확인/디버깅 명령

```zsh
# 장치 나열
v4l2-ctl --list-devices
ls -l /dev/video* /dev/v4l/by-path

# 포맷 확인
v4l2-ctl --device=/dev/video0 --list-formats-ext

# 현재 컨트롤 확인
v4l2-ctl -d /dev/video0 --all

# GStreamer 플러그인 확인
gst-inspect-1.0 x264enc | head
gst-inspect-1.0 v4l2src | head

# ALSA
arecord -l
arecord -L
```

---

### 참고/주의

* OpenCV **시스템 빌드**를 쓰기 위해 비전 venv는 `--system-site-packages`로 생성.
* MediaPipe는 CPU 사용량이 큼. 시작은 `--w 960 --h 540 --fps 20 --complexity 0` 추천.
* Windows 수신 지연은 `rtpjitterbuffer latency=40~60`에서 조정.
* Dynamixel: 토크 ON 여부 확인 후 목표각 전송. 안전을 위해 각도/속도 제한을 보수적으로.
* 카메라 컨트롤은 파이프라인 포맷 전환 시 초기화될 수 있어 **스트리밍 직전** 재적용 권장.

---