# Jetson Orin Nano 초기 세팅 & 장치 설정 정리

이 문서는 Jetson Orin Nano를 부팅한 뒤 개발 환경을 세팅하고,
카메라, 마이크, 아두이노, Dynamixel 모터를 사용할 수 있도록 한 과정을 정리한 것입니다.

---

## 1. Jetson Orin Nano 기본 세팅

1. JetPack 최신 버전이 설치된 상태에서 부팅
2. Wi-Fi / 유선 인터넷 연결 확인
3. 시스템 업데이트
   ```bash
   sudo apt update && sudo apt upgrade -y
   ```
4. 개발 툴 설치
   ```bash
   sudo apt install -y git curl wget build-essential cmake pkg-config        python3-dev python3-venv python3-pip        v4l-utils alsa-utils ffmpeg libopencv-dev
   ```

---

## 2. Python 가상환경 설정

1. 프로젝트 폴더 생성 및 이동
   ```bash
   mkdir -p ~/project-2 && cd ~/project-2
   ```

2. 가상환경 생성 및 활성화
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. 필수 패키지 설치
   ```bash
   pip install --upgrade pip
   pip install opencv-python sounddevice dynamixel-sdk numpy
   ```

---

## 3. 오디오 테스트

마이크 입력과 스피커 출력을 확인하기 위해 ALSA 유틸리티 사용.

- 녹음 및 재생 테스트
  ```bash
  arecord -D plughw:<카드번호>,0 -f cd -d 3 test.wav
  aplay test.wav
  ```

- 연결된 장치 확인
  ```bash
  arecord -l
  ```

---

## 4. VS Code + SSH 개발환경

1. Jetson에 SSH 서버 설치
   ```bash
   sudo apt install openssh-server -y
   ```

2. PC에서 VS Code 실행 → `Remote-SSH` 확장 설치  
   → Command Palette (`Ctrl+Shift+P`) → `Remote-SSH: Connect to Host`

3. Jetson 주소 입력 후 접속

---

## 5. PlatformIO (Arduino 펌웨어 개발)

1. PlatformIO 설치 (로컬 PC 또는 Jetson)
   ```bash
   pip install platformio
   ```

2. 아두이노 보드 라이브러리 설치
   ```bash
   lib_deps =
   TaskScheduler
   nicohood/PinChangeInterrupt@^1.2.9
   adafruit/Adafruit NeoPixel@^1.15.1
   ```

3. 보드 연결 확인
   ```bash
   pio device list
   ```

4. 포트 권한 문제 해결 (dialout 그룹)
   ```bash
   sudo usermod -aG dialout $USER
   newgrp dialout
   ```

---

## 6. udev 규칙 설정

장치 이름이 매번 바뀌지 않도록 고정하기 위해 udev 규칙을 작성.

### (1) Arduino 보드
`/etc/udev/rules.d/99-arduino.rules`
```udev
SUBSYSTEM=="tty", ATTRS{idVendor}=="2341", ATTRS{idProduct}=="0043", SYMLINK+="arduino", MODE="0666", GROUP="dialout"
```

### (2) 카메라 (왼쪽 / 오른쪽)
왼쪽 카메라(`/dev/cam_left`)
```udev
SUBSYSTEM=="video4linux", ATTRS{idVendor}=="2ce3", ATTRS{idProduct}=="c670", KERNEL=="video*", SYMLINK+="cam_left", MODE="0666"
```

오른쪽 카메라(`/dev/cam_right`)
```udev
SUBSYSTEM=="video4linux", ATTRS{idVendor}=="2ce3", ATTRS{idProduct}=="c670", KERNEL=="video*", SYMLINK+="cam_right", MODE="0666"
```

### (3) Follower Arm
```udev
SUBSYSTEM=="tty", ATTRS{idVendor}=="1a86", ATTRS{idProduct}=="55d3", ATTRS{serial}=="5970073211", SYMLINK+="follower_arm", MODE="0666", GROUP="dialout"
```

### (4) Leader Arm
```udev
SUBSYSTEM=="tty", ATTRS{idVendor}=="1a86", ATTRS{idProduct}=="55d3", ATTRS{serial}=="5970073130", SYMLINK+="leader_arm", MODE="0666", GROUP="dialout"
```

### (5) USB 마이크 (Blue Tiki)
```udev
SUBSYSTEM=="sound", ATTRS{idVendor}=="b58e", ATTRS{idProduct}=="8454", ATTRS{serial}=="2012529", SYMLINK+="mic_blue", MODE="0666", GROUP="audio"
```

적용:
```bash
sudo udevadm control --reload-rules && sudo udevadm trigger
```
---

## 8. 장치 요약

- `/dev/arduino` → Arduino 보드
- `/dev/cam_left` → 왼쪽 카메라
- `/dev/cam_right` → 오른쪽 카메라
- `/dev/follower_arm` → Follower Arm
- `/dev/leader_arm` → Leader Arm
- `/dev/mic_blue` → USB Blue Tiki 마이크

---