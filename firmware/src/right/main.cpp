#include <Arduino.h>
#include <TaskScheduler.h>
#include <Wire.h>
#include <PinChangeInterrupt.h>
#include <Adafruit_NeoPixel.h>
#include <math.h>

// ============================================================================
// MPU-6050 (Accelerometer) Configuration
// ============================================================================

// MPU6050 registers and constants
constexpr uint8_t MPU_ADDR       = 0x68;
constexpr uint8_t REG_PWR_MGMT_1 = 0x6B;
constexpr uint8_t REG_ACCEL_CFG  = 0x1C;
constexpr uint8_t REG_CONFIG     = 0x1A;
constexpr uint8_t REG_SMPLRT_DIV = 0x19;
constexpr uint8_t REG_ACCEL_XOUT = 0x3B;

// Acceleration scale (LSB/g) - +/-4g range
constexpr float   ACC_LSB_PER_G  = 8192.0f;
constexpr long    ONE_G_LSB      = (long)ACC_LSB_PER_G;

// Sensor internal filter/sample rate
constexpr uint8_t CONFIG_DLPF    = 0x03;      // ~44Hz
constexpr uint8_t SMPLRT_DIV_VAL = 0x03;      // 1kHz/(1+3)=250Hz

// Calibration settings
constexpr uint16_t CALI_SAMPLES  = 500;
constexpr uint16_t CALI_DELAY_MS = 5;

// Main loop sample rate
constexpr uint16_t LOOP_DELAY_MS = 2;

// Output interval
const uint16_t PRINT_MS = 300;

// EMA coefficient
const float EMA_ALPHA = 0.2f;

// Peak trigger/hold
float TRIGGER_G = 0.35f;
float RELEASE_G = 0.25f;
float PEAK_EPS  = 0.02f;
const unsigned long PEAK_HOLD_MS = 3000;

// ============================================================================
// NeoPixel / LED Configuration
// ============================================================================

// 핀 및 상수 정의
const int LED_BUTTON_PIN = 2;     // 밝기 조절 버튼
const int ROBOT_BUTTON_PIN = 3;   // 로봇팔 제어 버튼
const int NEOPIXEL_PIN = 4;
const int NUM_PIXELS = 16;
const int MAX_BRIGHTNESS = 255;
const int BRIGHTNESS_STEPS = 5;
const int BRIGHTNESS_UNIT = MAX_BRIGHTNESS / BRIGHTNESS_STEPS;
const int DEBOUNCE_DELAY = 200;

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

// Neopixel 설정
Adafruit_NeoPixel pixels(NUM_PIXELS, NEOPIXEL_PIN, NEO_RGBW + NEO_KHZ800);

// ============================================================================
// MPU-6050 상태 변수
// ============================================================================

long ax_off = 0, ay_off = 0, az_off = 0;
float ax_ema = 0.0f, ay_ema = 0.0f, az_ema = 0.0f;
float amag_ema = 0.0f, adyn_ema = 0.0f;
float peak_g = 0.0f;
unsigned long peak_t = 0;
bool triggered = false;
bool emaInitialized = false;
bool freezeOutput = false;
unsigned long lastPrint = 0;

float noise_mean = 0.0f;
float noise_std  = 0.0f;

// 함수 프로토타입 선언
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
void mpuInit();
void calibrateAccel();
void calibrateNoiseFloor();
void processMPU6050Task();

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

// 태스크 스케줄러
Scheduler runner;

// 태스크 정의
Task tDisplayStatus(STATUS_INTERVAL, TASK_FOREVER, &displayStatusTask);
Task tProcessButtons(50, TASK_FOREVER, &processButtonsTask);
Task tUpdateNeopixel(50, TASK_FOREVER, &updateNeopixelTask);  // 더 부드러운 LED 효과를 위해 50ms로 단축
Task tProcessSerial(100, TASK_FOREVER, &processSerialTask);
Task tHeartbeat(HEARTBEAT_INTERVAL, TASK_FOREVER, &heartbeatTask);
Task tProcessMPU6050(LOOP_DELAY_MS, 0, &processMPU6050Task);

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

void setup() {
  Serial.begin(9600);
  Serial.println("=== Arduino NeoPixel Controller Started ===");
  Serial.println("Version: 2.0");
  Serial.println("Voice Command Ready");

  pinMode(LED_BUTTON_PIN, INPUT_PULLUP);
  pinMode(ROBOT_BUTTON_PIN, INPUT_PULLUP);
  
  attachPCINT(digitalPinToPCINT(LED_BUTTON_PIN), ledButtonInterrupt, FALLING);
  attachPCINT(digitalPinToPCINT(ROBOT_BUTTON_PIN), robotButtonInterrupt, FALLING);

  pixels.begin();
  pixels.clear();
  pixels.show();

  Wire.begin();
  Wire.setClock(400000);
  //mpuInit();
  //calibrateAccel();
  //calibrateNoiseFloor();

  // 시작 시 LED 효과 제거 (바로 꺼진 상태로 시작)
  startupEffect();

  runner.init();
  runner.addTask(tDisplayStatus);
  runner.addTask(tProcessButtons);
  runner.addTask(tUpdateNeopixel);
  runner.addTask(tProcessSerial);
  runner.addTask(tHeartbeat);
  runner.addTask(tProcessMPU6050);
  
  tDisplayStatus.enable();
  tProcessButtons.enable();
  tUpdateNeopixel.enable();
  tProcessSerial.enable();
  tHeartbeat.enable();
  tProcessMPU6050.enable();

  // 초기 상태 전송
  sendStatus();
  Serial.println("Arduino Ready - Voice Commands Enabled");
}

// ============================================================================
// MPU-6050 유틸리티 함수
// ============================================================================

static int16_t readWord(uint8_t regH) {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(regH);
  Wire.endTransmission(false);
  Wire.requestFrom(MPU_ADDR, (uint8_t)2);
  while (Wire.available() < 2) {
    ;
  }
  int16_t hi = Wire.read();
  int16_t lo = Wire.read();
  return (hi << 8) | lo;
}

static void writeReg8(uint8_t reg, uint8_t val) {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(reg);
  Wire.write(val);
  Wire.endTransmission();
}

void mpuInit() {
  writeReg8(REG_PWR_MGMT_1, 0x00);
  delay(100);
  writeReg8(REG_CONFIG, CONFIG_DLPF);
  writeReg8(REG_SMPLRT_DIV, SMPLRT_DIV_VAL);
  writeReg8(REG_ACCEL_CFG, 0x08);  // +/-4g
  delay(100);
}

void calibrateAccel() {
  long sax = 0, say = 0, saz = 0;

  Serial.println(F("\n[MPU] Hold still for calibration (approx 3s)..."));
  delay(500);

  for (uint16_t i = 0; i < CALI_SAMPLES; ++i) {
    int16_t ax = readWord(REG_ACCEL_XOUT + 0);
    int16_t ay = readWord(REG_ACCEL_XOUT + 2);
    int16_t az = readWord(REG_ACCEL_XOUT + 4);
    sax += ax;
    say += ay;
    saz += az;
    delay(CALI_DELAY_MS);
  }

  ax_off = sax / (long)CALI_SAMPLES;
  ay_off = say / (long)CALI_SAMPLES;
  az_off = saz / (long)CALI_SAMPLES - ONE_G_LSB;

  Serial.print(F("[MPU] Offsets ax=")); Serial.print(ax_off);
  Serial.print(F(" ay=")); Serial.print(ay_off);
  Serial.print(F(" az=")); Serial.println(az_off);
}

void calibrateNoiseFloor() {
  const int sampleCount = 250;
  float sum = 0.0f;
  float sumSq = 0.0f;

  for (int i = 0; i < sampleCount; ++i) {
    int16_t ax = readWord(REG_ACCEL_XOUT + 0);
    int16_t ay = readWord(REG_ACCEL_XOUT + 2);
    int16_t az = readWord(REG_ACCEL_XOUT + 4);

    float axg = (float)((long)ax - ax_off) / ACC_LSB_PER_G;
    float ayg = (float)((long)ay - ay_off) / ACC_LSB_PER_G;
    float azg = (float)((long)az - az_off) / ACC_LSB_PER_G;

    float amag = sqrtf(axg * axg + ayg * ayg + azg * azg);
    float adyn = amag - 1.0f;
    if (adyn < 0.0f) adyn = 0.0f;

    sum   += adyn;
    sumSq += adyn * adyn;
    delay(4);
  }

  noise_mean = sum / sampleCount;
  float variance = (sumSq / sampleCount) - (noise_mean * noise_mean);
  noise_std = variance > 0.0f ? sqrtf(variance) : 0.0f;

  TRIGGER_G = max(0.35f, noise_mean + 6.0f * noise_std);
  RELEASE_G = max(0.25f, noise_mean + 4.0f * noise_std);
  PEAK_EPS  = max(0.02f, 3.0f * noise_std);

  Serial.print(F("[MPU] Noise mean=")); Serial.print(noise_mean, 4);
  Serial.print(F(" std=")); Serial.print(noise_std, 4);
  Serial.print(F(" TRG=")); Serial.print(TRIGGER_G, 3);
  Serial.print(F(" REL=")); Serial.print(RELEASE_G, 3);
  Serial.print(F(" EPS=")); Serial.println(PEAK_EPS, 3);
}

void processMPU6050Task() {
  int16_t ax_raw = readWord(REG_ACCEL_XOUT + 0);
  int16_t ay_raw = readWord(REG_ACCEL_XOUT + 2);
  int16_t az_raw = readWord(REG_ACCEL_XOUT + 4);

  long ax_corr = (long)ax_raw - ax_off;
  long ay_corr = (long)ay_raw - ay_off;
  long az_corr = (long)az_raw - az_off;

  float ax_g = (float)ax_corr / ACC_LSB_PER_G;
  float ay_g = (float)ay_corr / ACC_LSB_PER_G;
  float az_g = (float)az_corr / ACC_LSB_PER_G;

  float amag = sqrtf(ax_g * ax_g + ay_g * ay_g + az_g * az_g);
  float adyn = amag - 1.0f;
  if (adyn < 0.0f) adyn = 0.0f;

  float a_for_peak = fabsf(adyn);

  if (!triggered && a_for_peak >= TRIGGER_G) {
    triggered = true;
    Serial.print(F("TRG,")); Serial.print(millis());
    Serial.print(F(",")); Serial.println(a_for_peak, 4);
  } else if (triggered && a_for_peak <= RELEASE_G) {
    triggered = false;
  }

  if (a_for_peak >= TRIGGER_G && (a_for_peak > peak_g + PEAK_EPS)) {
    peak_g = a_for_peak;
    peak_t = millis();
    Serial.print(F("PEAK,")); Serial.print(peak_t);
    Serial.print(F(",")); Serial.print(peak_g, 4);
    Serial.println(F("g"));
  }

  if (peak_t && (millis() - peak_t > PEAK_HOLD_MS)) {
    peak_g *= 0.95f;
    if (peak_g < 0.02f) peak_g = 0.0f;
  }

  if (!emaInitialized) {
    ax_ema = ax_g;
    ay_ema = ay_g;
    az_ema = az_g;
    amag_ema = amag;
    adyn_ema = adyn;
    emaInitialized = true;
  } else if (!freezeOutput) {
    ax_ema   = EMA_ALPHA * ax_g   + (1.0f - EMA_ALPHA) * ax_ema;
    ay_ema   = EMA_ALPHA * ay_g   + (1.0f - EMA_ALPHA) * ay_ema;
    az_ema   = EMA_ALPHA * az_g   + (1.0f - EMA_ALPHA) * az_ema;
    amag_ema = EMA_ALPHA * amag   + (1.0f - EMA_ALPHA) * amag_ema;
    adyn_ema = EMA_ALPHA * adyn   + (1.0f - EMA_ALPHA) * adyn_ema;
  }

  unsigned long now = millis();
  if (!freezeOutput && now - lastPrint >= PRINT_MS) {
    lastPrint = now;
    Serial.print(F("ACC,"));
    Serial.print(now); Serial.print(',');
    Serial.print(ax_ema, 4); Serial.print(',');
    Serial.print(ay_ema, 4); Serial.print(',');
    Serial.print(az_ema, 4); Serial.print(',');
    Serial.print(amag_ema, 4); Serial.print(',');
    Serial.print(adyn_ema, 4); Serial.print(',');
    Serial.println(peak_g, 4);
  }
}

void startupEffect() {
  // 시작 시 무지개 효과
  for(int j = 0; j < 256; j++) {
    for(int i = 0; i < NUM_PIXELS; i++) {
      pixels.setPixelColor(i, Wheel((i * 256 / NUM_PIXELS + j) & 255));
    }
    pixels.show();
    delay(5);
  }
  pixels.clear();
  pixels.show();
}

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
    // 음성 명령 처리 (짧은 명령어)
    else if (command == "ON" || command == "LIGHT_ON") {
      isLightOn = true;
      if (brightnessLevel == 0) {
        applyBrightnessLevel(BRIGHTNESS_STEPS);
      } else {
        applyBrightnessLevel(brightnessLevel);
      }
      Serial.println(F("OK: Light ON"));
    }
    else if (command == "OFF" || command == "LIGHT_OFF") {
      isLightOn = false;
      Serial.println(F("OK: Light OFF"));
    }
    else if (command == "UP" || command == "BRIGHTNESS_UP") {
      if (adjustBrightness(1, F("OK: Brightness UP to "))) {
        isLightOn = true;
      }
    }
    else if (command == "DOWN" || command == "BRIGHTNESS_DOWN") {
      if (adjustBrightness(-1, F("OK: Brightness DOWN to "))) {
        if (brightnessLevel == 0) {
          isLightOn = false;
        }
      }
    }
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
      Serial.println(F("OK: Rainbow Effect"));
      triggerLedEffect(3);
    }
    else if (command == "FREEZE_ACC") {
      freezeOutput = true;
      Serial.println(F("ACC:PAUSED"));
    }
    else if (command == "RESUME_ACC") {
      freezeOutput = false;
      Serial.println(F("ACC:RESUMED"));
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
        if (elapsed < 3000) {
          for(int i = 0; i < NUM_PIXELS; i++) {
            int hue = (i * 256 / NUM_PIXELS + (elapsed / 10)) % 256;
            uint32_t color = Wheel(hue);
            // 밝기 조절
            int r = ((color >> 16) & 0xFF) * currentBrightness / 255;
            int g = ((color >> 8) & 0xFF) * currentBrightness / 255;
            int b = (color & 0xFF) * currentBrightness / 255;
            pixels.setPixelColor(i, pixels.Color(g, r, b));
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

      case 8: // 웨이크워드 감지 - 빠른 파란색 깜빡임
        if (elapsed < 600) {
          // 150ms 주기로 4번 깜빡임 (150ms on, 150ms off 반복)
          int blinkCycle = (elapsed / 300) % 2;
          if (blinkCycle == 0) {
            // 켜짐 - 파란색
            int brightness = max(currentBrightness, 255);  // 최소 밝기 보장
            setAllPixels(0, brightness, 0);
          } else {
            // 꺼짐
            setAllPixels(0, 0, 0);
          }
        } else {
          ledEffectActive = false;
        }
        break;

      case 9: // 시작
        if (elapsed < 600) {
          // 150ms 주기로 4번 깜빡임 (150ms on, 150ms off 반복)
          int blinkCycle = (elapsed / 300) % 2;
          if (blinkCycle == 0) {
            // 켜짐 - 파란색
            int brightness = max(currentBrightness, 255);  // 최소 밝기 보장
            setAllPixels(0, 0, brightness);
          } else {
            // 꺼짐
            setAllPixels(0, 0, 0);
          }
        } else {
          ledEffectActive = false;
        }
        break;

      case 10: // 집으로
        if (elapsed < 600) {
          // 150ms 주기로 4번 깜빡임 (150ms on, 150ms off 반복)
          int blinkCycle = (elapsed / 300) % 2;
          if (blinkCycle == 0) {
            // 켜짐 - 파란색
            int brightness = max(currentBrightness, 255);  // 최소 밝기 보장
            setAllPixels(brightness, brightness, 0);
          } else {
            // 꺼짐
            setAllPixels(0, 0, 0);
          }
        } else {
          ledEffectActive = false;
        }
        break;

      case 11: // 정지
        if (elapsed < 600) {
          // 150ms 주기로 4번 깜빡임 (150ms on, 150ms off 반복)
          int blinkCycle = (elapsed / 300) % 2;
          if (blinkCycle == 0) {
            // 켜짐 - 파란색
            int brightness = max(currentBrightness, 255);  // 최소 밝기 보장
            setAllPixels(brightness, 0, 0);
          } else {
            // 꺼짐
            setAllPixels(0, 0, 0);
          }
        } else {
          ledEffectActive = false;
        }
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

  // 연결 상태 표시 제거 (깜빡임 없음)
  // if (!systemConnected && NUM_PIXELS > 0) {
  //   // 연결 안됨 - 빨간색 깜빡임
  //   int redIntensity = (millis() % 1000 < 500) ? 50 : 0;
  //   pixels.setPixelColor(0, pixels.Color(redIntensity, 0, 0));
  // }

  pixels.show();
}

void setAllPixels(int r, int g, int b) {
  for (int i = 0; i < NUM_PIXELS; i++) {
    pixels.setPixelColor(i, pixels.Color(g, r, b));
  }
}

// 무지개 색상 생성 함수
uint32_t Wheel(byte WheelPos) {
  WheelPos = 255 - WheelPos;
  if(WheelPos < 85) {
    return pixels.Color(255 - WheelPos * 3, 0, WheelPos * 3);
  }
  if(WheelPos < 170) {
    WheelPos -= 85;
    return pixels.Color(0, WheelPos * 3, 255 - WheelPos * 3);
  }
  WheelPos -= 170;
  return pixels.Color(WheelPos * 3, 255 - WheelPos * 3, 0);
}

void loop() {
  runner.execute();
}
