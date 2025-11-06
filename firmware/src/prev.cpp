// MARK: - Includes

#include <Arduino.h>
#include <TaskScheduler.h>
#include <PinChangeInterrupt.h>
#include <Adafruit_NeoPixel.h>
#include <math.h>

// MARK: - LED Configuration

// 핀 및 상수 정의
const int LED_BUTTON_PIN = 2;     // 밝기 조절 버튼
const int ROBOT_BUTTON_PIN = 3;   // 로봇팔 제어 버튼
const int NEOPIXEL_PIN = 4;       // 기존 16개 LED 스트립
const int NEOPIXEL_PIN_RING = 5;  // 새로운 8개 LED 링
const int NUM_PIXELS = 16;        // 기존 스트립 LED 개수
const int NUM_PIXELS_RING = 8;    // 링 LED 개수
const int NUM_STRIPS = 2;         // NeoPixel 스트립 개수
const int MAX_BRIGHTNESS = 255;
const int BRIGHTNESS_STEPS = 5;
const int BRIGHTNESS_UNIT = MAX_BRIGHTNESS / BRIGHTNESS_STEPS;
const int DEBOUNCE_DELAY = 200;

// MARK: - State Variables

// 상태 변수
volatile int brightnessLevel = 0;
volatile bool ledButtonPressed = false;
volatile bool robotButtonPressed = false;
volatile unsigned long lastLedButtonTime = 0;
volatile unsigned long lastRobotButtonTime = 0;
volatile int robotMode = 0;         // 로봇 모드 (0 또는 1)
int currentBrightness = 0;
unsigned long lastStatusSend = 0;
const unsigned long STATUS_INTERVAL = 1000;
bool systemConnected = false;
unsigned long lastHeartbeat = 0;
const unsigned long HEARTBEAT_INTERVAL = 5000;

// LED 효과 변수
bool ledEffectActive = false;
unsigned long ledEffectStart = 0;
int effectType = 0;

// 색상 변수
int currentR = 255, currentG = 255, currentB = 255; // 현재 색상 (RGB)
bool isLightOn = true; // 조명 상태

// MARK: - NeoPixel Setup

// Neopixel 설정 - 배열로 관리
const int pixelCounts[NUM_STRIPS] = {NUM_PIXELS, NUM_PIXELS_RING};
Adafruit_NeoPixel pixels[NUM_STRIPS] = {
  Adafruit_NeoPixel(NUM_PIXELS, NEOPIXEL_PIN, NEO_RGBW + NEO_KHZ800),      // 기존 16개 스트립
  Adafruit_NeoPixel(NUM_PIXELS_RING, NEOPIXEL_PIN_RING, NEO_RGB + NEO_KHZ800)  // 새 8개 링
};

// MARK: - Function Prototypes

// 함수 프로토타입
void displayStatusTask();
void processButtonsTask();
void updateNeopixelTask();
void processSerialTask();
void heartbeatTask();
void ledButtonInterrupt();
void robotButtonInterrupt();
void startupEffect();
void sendStatus();
void triggerLedEffect(int type);
void setAllPixels(int r, int g, int b);
uint32_t Wheel(byte WheelPos);
void blinkEffect(unsigned long elapsed, int r, int g, int b);

// MARK: - Helper Functions

static void applyBrightnessLevel(int level, const __FlashStringHelper* feedback = nullptr) {
  int constrained = constrain(level, 0, BRIGHTNESS_STEPS);
  brightnessLevel = constrained;
  currentBrightness = brightnessLevel * BRIGHTNESS_UNIT;
  isLightOn = brightnessLevel > 0;

  if (feedback) {
    Serial.print(feedback);
    Serial.println(brightnessLevel);
  }
}

static bool adjustBrightness(int delta, const __FlashStringHelper* feedback = nullptr) {
  if (delta == 0) {
    return false;
  }

  int target = constrain(brightnessLevel + delta, 0, BRIGHTNESS_STEPS);
  if (target == brightnessLevel) {
    return false;
  }

  applyBrightnessLevel(target, feedback);
  return true;
}

static void setColorPreset(uint8_t r, uint8_t g, uint8_t b, int effect, const __FlashStringHelper* message = nullptr) {
  currentR = r;
  currentG = g;
  currentB = b;
  isLightOn = true;

  if (message) {
    Serial.println(message);
  }

  if (effect >= 0) {
    triggerLedEffect(effect);
  }
}

// MARK: - Task Scheduler

// 태스크 스케줄러
Scheduler runner;

// 태스크 정의
Task tDisplayStatus(STATUS_INTERVAL, TASK_FOREVER, &displayStatusTask);
Task tProcessButtons(50, TASK_FOREVER, &processButtonsTask);
Task tUpdateNeopixel(50, TASK_FOREVER, &updateNeopixelTask);  // 더 부드러운 LED 효과를 위해 50ms로 단축
Task tProcessSerial(100, TASK_FOREVER, &processSerialTask);
Task tHeartbeat(HEARTBEAT_INTERVAL, TASK_FOREVER, &heartbeatTask);

// MARK: - Button Interrupts

// 버튼 인터럽트 핸들러 (디바운싱 포함)
void ledButtonInterrupt() {
  unsigned long currentTime = millis();
  if (currentTime - lastLedButtonTime > DEBOUNCE_DELAY) {
    ledButtonPressed = true;
    lastLedButtonTime = currentTime;
  }
}

void robotButtonInterrupt() {
  unsigned long currentTime = millis();
  if (currentTime - lastRobotButtonTime > DEBOUNCE_DELAY) {
    robotButtonPressed = true;
    lastRobotButtonTime = currentTime;
  }
}

// MARK: - Setup

void setup() {
  Serial.begin(9600);
  Serial.println("=== Arduino NeoPixel Controller Started ===");
  Serial.println("Version: 2.0");
  Serial.println("Voice Command Ready");

  pinMode(LED_BUTTON_PIN, INPUT_PULLUP);
  pinMode(ROBOT_BUTTON_PIN, INPUT_PULLUP);

  attachPCINT(digitalPinToPCINT(LED_BUTTON_PIN), ledButtonInterrupt, FALLING);
  attachPCINT(digitalPinToPCINT(ROBOT_BUTTON_PIN), robotButtonInterrupt, FALLING);

  // 모든 NeoPixel 스트립 초기화
  for (int i = 0; i < NUM_STRIPS; i++) {
    pixels[i].begin();
    pixels[i].clear();
    pixels[i].show();
  }

  // 시작 시 LED 효과 제거 (바로 꺼진 상태로 시작)
  startupEffect();

  runner.init();
  runner.addTask(tDisplayStatus);
  runner.addTask(tProcessButtons);
  runner.addTask(tUpdateNeopixel);
  runner.addTask(tProcessSerial);
  runner.addTask(tHeartbeat);
  
  tDisplayStatus.enable();
  tProcessButtons.enable();
  tUpdateNeopixel.enable();
  tProcessSerial.enable();
  tHeartbeat.enable();

  // 초기 상태 전송
  sendStatus();
  Serial.println("Arduino Ready - Voice Commands Enabled");
}

// MARK: - Startup Effects

void startupEffect() {
  // 시작 시 무지개 효과 - 모든 스트립에 적용
  for(int j = 0; j < 256; j++) {
    for(int strip = 0; strip < NUM_STRIPS; strip++) {
      int numLeds = pixelCounts[strip];
      for(int i = 0; i < numLeds; i++) {
        pixels[strip].setPixelColor(i, Wheel((i * 256 / numLeds + j) & 255));
      }
      pixels[strip].show();
    }
    delay(5);
  }
  // 모든 스트립 끄기
  for(int strip = 0; strip < NUM_STRIPS; strip++) {
    pixels[strip].clear();
    pixels[strip].show();
  }
}

// MARK: - Task Functions

void displayStatusTask() {
  if (millis() - lastStatusSend > STATUS_INTERVAL) {
    Serial.print(F("INFO: B"));
    Serial.print(brightnessLevel);
    Serial.print(F("/"));
    Serial.print(BRIGHTNESS_STEPS);
    Serial.print(F(" ("));
    Serial.print(currentBrightness);
    Serial.print(F(") M"));
    Serial.print(robotMode);
    Serial.print(F(" C"));
    Serial.print(systemConnected ? F("1") : F("0"));
    Serial.print(F(" E"));
    Serial.println(effectType);
    lastStatusSend = millis();
  }
}

void processButtonsTask() {
  // LED 밝기 조절 버튼 처리
  if (ledButtonPressed) {
    int nextLevel = (brightnessLevel + 1) % (BRIGHTNESS_STEPS + 1);
    applyBrightnessLevel(nextLevel);
    ledButtonPressed = false;

    Serial.print(F("BTN: LED Level "));
    Serial.print(brightnessLevel);
    Serial.print(F(" Brightness "));
    Serial.println(currentBrightness);
    
    // LED 밝기 명령 전송 (Python 시스템이 필요하다면)
    Serial.print(F("CMD:LED:"));
    Serial.println(brightnessLevel);
    
    // LED 효과 트리거
    sendStatus();
  }
  
  // 로봇 제어 버튼 처리
  if (robotButtonPressed) {
    robotMode = !robotMode;  // 0과 1 사이 토글
    robotButtonPressed = false;
    
    Serial.print(F("BTN: Robot Mode "));
    Serial.println(robotMode);
    
    // 로봇 제어 명령 전송 (Python 시스템에서 수신)
    Serial.print(F("CMD:ROBOT:"));
    Serial.println(robotMode);
    
    // LED 효과 트리거
    triggerLedEffect(robotMode == 0 ? 2 : 3); // 모드에 따른 다른 효과
    sendStatus();
  }
}

void processSerialTask() {
  if (Serial.available()) {
    String command = Serial.readStringUntil('\n');
    command.trim();
    
    if (command.startsWith("HEARTBEAT")) {
      systemConnected = true;
      lastHeartbeat = millis();
      Serial.println(F("ACK:HEARTBEAT"));
      sendStatus();  // Heartbeat 응답과 함께 상태도 전송
    }
    else if (command.startsWith("STATUS")) {
      Serial.println(F("ACK:STATUS"));  // STATUS 명령 수신 확인
      sendStatus();
    }
    // 조명 ON/OFF
    else if (command == "ON" || command == "LIGHT_ON") {
      isLightOn = true;
      applyBrightnessLevel(brightnessLevel == 0 ? BRIGHTNESS_STEPS : brightnessLevel);
      Serial.println(F("OK: Light ON"));
    }
    else if (command == "OFF" || command == "LIGHT_OFF") {
      isLightOn = false;
      Serial.println(F("OK: Light OFF"));
    }
    // 밝기 조절
    else if (command == "UP" || command == "BRIGHTNESS_UP") {
      if (adjustBrightness(1, F("OK: Brightness UP to "))) {
        isLightOn = true;
      }
    }
    else if (command == "DOWN" || command == "BRIGHTNESS_DOWN") {
      if (adjustBrightness(-1, F("OK: Brightness DOWN to "))) {
        isLightOn = (brightnessLevel > 0);
      }
    }
    // 색상 명령
    else if (command == "R" || command == "RED" || command == "COLOR_RED") {
      setColorPreset(255, 0, 0, 4, F("OK: Color RED"));
    }
    else if (command == "G" || command == "GREEN" || command == "COLOR_GREEN") {
      setColorPreset(0, 255, 0, 6, F("OK: Color GREEN"));
    }
    else if (command == "B" || command == "BLUE" || command == "COLOR_BLUE") {
      setColorPreset(0, 0, 255, 5, F("OK: Color BLUE"));
    }
    else if (command == "Y" || command == "YELLOW" || command == "COLOR_YELLOW") {
      setColorPreset(255, 255, 0, 7, F("OK: Color YELLOW"));
    }
    else if (command == "W" || command == "WHITE" || command == "COLOR_WHITE") {
      setColorPreset(255, 255, 255, -1, F("OK: Color WHITE"));
    }
    else if (command == "RAINBOW" || command == "COLOR_RAINBOW") {
      isLightOn = true;
      triggerLedEffect(3);
      Serial.println(F("OK: Rainbow Effect"));
    }
    // 기존 명령들
    else if (command.startsWith("SET_BRIGHTNESS:")) {
      int newLevel = command.substring(15).toInt();
      if (newLevel >= 0 && newLevel <= BRIGHTNESS_STEPS) {
        applyBrightnessLevel(newLevel, F("Brightness set to level: "));
        triggerLedEffect(1);
      }
    }
    else if (command.startsWith("SET_MODE:")) {
      int newMode = command.substring(9).toInt();
      if (newMode == 0 || newMode == 1) {
        robotMode = newMode;
        Serial.print(F("Robot mode set to: "));
        Serial.println(robotMode);
        triggerLedEffect(newMode == 0 ? 2 : 3);
      }
    }
    else if (command.startsWith("LED_EFFECT:")) {
      int effect = command.substring(11).toInt();
      triggerLedEffect(effect);
    }
    else if (command == "RESET") {
      // 시스템 리셋
      applyBrightnessLevel(0);
      robotMode = 0;
      triggerLedEffect(0);
      Serial.println(F("System Reset"));
      sendStatus();
    }
    else {
      Serial.print(F("ERR: Unknown command: "));
      Serial.println(command);
    }
  }
  
  // 연결 상태 확인 (5초 이상 heartbeat 없으면 연결 끊김으로 판단)
  if (systemConnected && (millis() - lastHeartbeat > HEARTBEAT_INTERVAL * 2)) {
    systemConnected = false;
    Serial.println(F("WARN: Connection timeout"));
  }
}

// MARK: - Heartbeat & Status

void heartbeatTask() {
  // 주기적으로 상태 전송
  sendStatus();
}

void sendStatus() {
  Serial.print(F("STATUS:BRIGHTNESS:"));
  Serial.print(brightnessLevel);
  Serial.print(F(":MODE:"));
  Serial.print(robotMode);
  Serial.print(F(":CONNECTED:"));
  Serial.print(systemConnected ? F("1") : F("0"));
  Serial.print(F(":LIGHT:"));
  Serial.print(isLightOn ? F("1") : F("0"));
  Serial.print(F(":COLOR:"));
  Serial.print(currentR);
  Serial.print(F(","));
  Serial.print(currentG);
  Serial.print(F(","));
  Serial.println(currentB);
}

// MARK: - LED Effects

void triggerLedEffect(int type) {
  effectType = type;
  ledEffectActive = true;
  ledEffectStart = millis();
}

void updateNeopixelTask() {
  if (ledEffectActive) {
    unsigned long elapsed = millis() - ledEffectStart;

    switch(effectType) {
      case 1: // 깜빡임 효과 (밝기 변경 시)
        if (elapsed < 300) {
          int flashBrightness = (elapsed % 200 < 100) ? currentBrightness : 0;
          int r = (currentR * flashBrightness) / 255;
          int g = (currentG * flashBrightness) / 255;
          int b = (currentB * flashBrightness) / 255;
          setAllPixels(r, g, b);
        } else {
          ledEffectActive = false;
        }
        break;

      case 2: // 파란색 페이드 (모드 0)
        if (elapsed < 1000) {
          int intensity = (sin((elapsed / 1000.0) * PI * 2) + 1) * currentBrightness / 2;
          setAllPixels(0, 0, intensity);
        } else {
          ledEffectActive = false;
        }
        break;

      case 3: // 무지개 효과
        if (elapsed < 500) {
          int brightness = max(currentBrightness, 255);  // 최소 밝기 보장
          // 모든 스트립에 무지개 효과 적용
          for(int strip = 0; strip < NUM_STRIPS; strip++) {
            int numLeds = pixelCounts[strip];
            for(int i = 0; i < numLeds; i++) {
              int hue = (i * 256 / numLeds + (elapsed / 10)) % 256;
              uint32_t color = Wheel(hue);
              // 밝기 조절
              int r = ((color >> 16) & 0xFF) * brightness / 255;
              int g = ((color >> 8) & 0xFF) * brightness / 255;
              int b = (color & 0xFF) * brightness / 255;
              pixels[strip].setPixelColor(i, pixels[strip].Color(g, r, b));
            }
          }
        } else {
          ledEffectActive = false;
        }
        break;

      case 4: // 빨간색 효과
      case 5: // 파란색 효과
      case 6: // 녹색 효과
      case 7: // 노란색 효과
        if (elapsed < 500) {
          // 색상 전환 효과
          float progress = elapsed / 500.0;
          int r = currentR * currentBrightness / 255 * progress;
          int g = currentG * currentBrightness / 255 * progress;
          int b = currentB * currentBrightness / 255 * progress;
          setAllPixels(r, g, b);
        } else {
          ledEffectActive = false;
        }
        break;

      case 8: // 웨이크워드 감지 - 파란색
        blinkEffect(elapsed, 0, max(currentBrightness, 255), 0);
        break;

      case 9: // 시작 - 파란색
        blinkEffect(elapsed, 0, max(currentBrightness, 255), 0);
        break;

      case 10: // 집으로 - 노란색
        blinkEffect(elapsed, max(currentBrightness, 255), max(currentBrightness, 255), 0);
        break;

      case 11: // 정지 - 빨간색
        blinkEffect(elapsed, max(currentBrightness, 255), 0, 0);
        break;

      default:
        ledEffectActive = false;
        break;
    }
  } else {
    // 일반 상태 - 현재 색상과 밝기로 표시
    if (isLightOn && currentBrightness > 0) {
      int r = (currentR * currentBrightness) / 255;
      int g = (currentG * currentBrightness) / 255;
      int b = (currentB * currentBrightness) / 255;
      setAllPixels(r, g, b);
    } else {
      setAllPixels(0, 0, 0); // 조명 꺼짐
    }
  }

  // 모든 스트립의 LED 표시 업데이트
  for(int strip = 0; strip < NUM_STRIPS; strip++) {
    pixels[strip].show();
  }
}

// MARK: - Effect Helpers

void blinkEffect(unsigned long elapsed, int r, int g, int b) {
  if (elapsed < 600) {
    int blinkCycle = (elapsed / 300) % 2;
    setAllPixels(blinkCycle == 0 ? r : 0, blinkCycle == 0 ? g : 0, blinkCycle == 0 ? b : 0);
  } else {
    ledEffectActive = false;
  }
}

void setAllPixels(int r, int g, int b) {
  // 모든 스트립의 모든 LED에 동일한 색상 적용
  for(int strip = 0; strip < NUM_STRIPS; strip++) {
    int numLeds = pixelCounts[strip];
    for (int i = 0; i < numLeds; i++) {
      pixels[strip].setPixelColor(i, pixels[strip].Color(g, r, b));
    }
  }
}

// MARK: - Color Utilities

// 무지개 색상 생성 함수
uint32_t Wheel(byte WheelPos) {
  WheelPos = 255 - WheelPos;
  if(WheelPos < 85) {
    return pixels[0].Color(255 - WheelPos * 3, 0, WheelPos * 3);
  }
  if(WheelPos < 170) {
    WheelPos -= 85;
    return pixels[0].Color(0, WheelPos * 3, 255 - WheelPos * 3);
  }
  WheelPos -= 170;
  return pixels[0].Color(WheelPos * 3, 255 - WheelPos * 3, 0);
}

// MARK: - Main Loop

void loop() {
  runner.execute();
}