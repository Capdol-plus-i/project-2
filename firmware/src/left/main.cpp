
#include <Wire.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

Adafruit_MPU6050 mpu;

// ===== 사용자 설정 =====
const unsigned long CALIB_MS      = 3000;     // 캘리브레이션 시간(정지 권장)
const unsigned long SAMPLE_DT_MS  = 10;       // 샘플 주기(ms) → 10 ms = 100 Hz
const unsigned long PRINT_DT_MS   = 100;      // 출력 주기(ms) → 10 Hz
const float         G_CONST       = 9.80665f; // m/s^2 → g 변환

// 떨림 대역(로봇팔 흔들림이 많은 구간)
const float HP_CUTOFF_HZ = 8.0f;              // 하한(자세/저주파 제거)
const float LP_CUTOFF_HZ = 40.0f;             // 상한(노이즈 제거)

// 점수(0~100) 매핑 기준 (g_rms: rad/s)
//   ~0.02: 거의 정지, 0.5 이상: 눈에 띄는 떨림
const float SCORE_S0   = 0.02f;               // 0점 기준
const float SCORE_S100 = 0.50f;               // 100점 기준

// ===== 내부 상태 =====
// 오프셋
float offAX=0, offAY=0, offAZ=0;              // accel(g)
float offGY=0, offGZ=0;                       // gyro (rad/s) — y(피치), z(요)

// 1차 HPF/LPF 계수
float hp_a=0.0f, lp_b=0.0f;

// 가속도 밴드패스 상태
float ahp_x=0, ahp_y=0, ahp_z=0;
float alp_x=0, alp_y=0, alp_z=0;
float prev_ax=0, prev_ay=0, prev_az=0;

// 자이로(y,z) 밴드패스 상태
float ghp_y=0, ghp_z=0;
float glp_y=0, glp_z=0;
float prev_gy=0, prev_gz=0;

// IIR RMS (제곱 평균의 지수평활), tau ≈ 1 s
float a_rms2_iir = 0.0f;   // aRMS^2 (g^2)
float g_rms2_iir = 0.0f;   // gRMS^2 ((rad/s)^2)
float rms_beta   = 0.0f;   // dt/(tau+dt)

unsigned long lastMs = 0, lastPrintMs = 0;

bool beginAuto() {
  if (mpu.begin(0x68)) return true;
  if (mpu.begin(0x69)) return true;
  return false;
}

// 자이로+가속도 캘리브레이션(정지 권장)
void calibrateSensors() {
  unsigned long start = millis();
  float sax=0, say=0, saz=0;
  float sgy=0, sgz=0;
  unsigned long n = 0;

  while (millis() - start < CALIB_MS) {
    sensors_event_t a, g, t;
    mpu.getEvent(&a, &g, &t);
    sax += a.acceleration.x / G_CONST;
    say += a.acceleration.y / G_CONST;
    saz += a.acceleration.z / G_CONST;
    sgy += g.gyro.y;
    sgz += g.gyro.z;
    n++;
    delay(3);
  }
  offAX = sax / n;                 // 정지 시 x≈0 g
  offAY = say / n;                 // 정지 시 y≈0 g
  offAZ = saz / n - 1.0f;          // 정지 시 z≈+1 g 되도록
  offGY = sgy / n;
  offGZ = sgz / n;

  // 상태 초기화
  ahp_x=ahp_y=ahp_z=0; alp_x=alp_y=alp_z=0; prev_ax=prev_ay=prev_az=0;
  ghp_y=ghp_z=0; glp_y=glp_z=0; prev_gy=prev_gz=0;
  a_rms2_iir = 0.0f; g_rms2_iir = 0.0f;
}

void setup() {
  Serial.begin(115200);
  delay(300);

  Wire.begin();
#if defined(TWBR) || ARDUINO >= 100
  Wire.setClock(400000); // Uno에서도 400 kHz 가능(배선 짧게)
#endif

  if (!beginAuto()) {
    Serial.println(F("[ERR] MPU6050 not found (0x68/0x69). 배선/I2C 확인."));
    while (1) { delay(500); }
  }

  // 센서 대역/범위: 100 Hz 샘플과 궁합
  mpu.setFilterBandwidth(MPU6050_BAND_44_HZ);
  mpu.setAccelerometerRange(MPU6050_RANGE_4_G);
  mpu.setGyroRange(MPU6050_RANGE_500_DEG);
  delay(150);

  // 필터 계수
  const float dt = SAMPLE_DT_MS / 1000.0f;
  const float tau_hp = 1.0f / (2.0f * M_PI * HP_CUTOFF_HZ);
  const float tau_lp = 1.0f / (2.0f * M_PI * LP_CUTOFF_HZ);
  hp_a = tau_hp / (tau_hp + dt);   // HPF: y[n]=a*(y[n-1]+x[n]-x[n-1])
  lp_b = dt / (tau_lp + dt);       // LPF: y[n]=y[n-1]+b*(x[n]-y[n-1])

  // IIR RMS 시간상수 ≈ 1 s
  const float tau_rms = 1.0f;
  rms_beta = dt / (tau_rms + dt);

  // 캘리브레이션
  Serial.print(F("[CAL] ")); Serial.print(CALIB_MS);
  Serial.println(F(" ms 동안 정지해 주세요..."));
  calibrateSensors();
  Serial.print(F("[CAL] accel offsets(g): "));
  Serial.print(offAX,3); Serial.print(F(", "));
  Serial.print(offAY,3); Serial.print(F(", "));
  Serial.println(offAZ,3);
  Serial.print(F("[CAL] gyro  offsets(rad/s): "));
  Serial.print(offGY,3); Serial.print(F(", "));
  Serial.println(offGZ,3);

  Serial.println(F("[READY] 흔들림 점수(0-100) / 등급 / gRMS(rad/s) / aRMS(g)"));
  Serial.println(F("time(ms) | score  grade   gRMS    aRMS"));
  Serial.println(F("--------------------------------------------------"));
}

void loop() {
  unsigned long now = millis();
  if (now - lastMs < SAMPLE_DT_MS) return;
  lastMs += SAMPLE_DT_MS;

  sensors_event_t a, g, t;
  mpu.getEvent(&a, &g, &t);

  // ----- (A) 가속도: g 단위로, 오프셋 제거 -----
  float ax = a.acceleration.x / G_CONST - offAX;
  float ay = a.acceleration.y / G_CONST - offAY;
  float az = a.acceleration.z / G_CONST - offAZ;

  // 1차 HPF → 1차 LPF (8–40 Hz)
  ahp_x = hp_a * (ahp_x + ax - prev_ax);
  ahp_y = hp_a * (ahp_y + ay - prev_ay);
  ahp_z = hp_a * (ahp_z + az - prev_az);
  prev_ax = ax; prev_ay = ay; prev_az = az;

  alp_x = alp_x + lp_b * (ahp_x - alp_x);
  alp_y = alp_y + lp_b * (ahp_y - alp_y);
  alp_z = alp_z + lp_b * (ahp_z - alp_z);

  float amag = sqrtf(alp_x*alp_x + alp_y*alp_y + alp_z*alp_z); // g

  // ----- (B) 자이로: rad/s, 오프셋 제거 -----
  float gy = g.gyro.y - offGY;  // pitch
  float gz = g.gyro.z - offGZ;  // yaw

  // 1차 HPF → 1차 LPF (8–40 Hz)
  ghp_y = hp_a * (ghp_y + gy - prev_gy);
  ghp_z = hp_a * (ghp_z + gz - prev_gz);
  prev_gy = gy; prev_gz = gz;

  glp_y = glp_y + lp_b * (ghp_y - glp_y);
  glp_z = glp_z + lp_b * (ghp_z - glp_z);

  float gmag = sqrtf(glp_y*glp_y + glp_z*glp_z);     // rad/s

  // ----- (C) IIR RMS 업데이트 -----
  a_rms2_iir = a_rms2_iir + rms_beta * (amag*amag - a_rms2_iir);
  g_rms2_iir = g_rms2_iir + rms_beta * (gmag*gmag - g_rms2_iir);
  float a_rms = sqrtf(a_rms2_iir);
  float g_rms = sqrtf(g_rms2_iir);

  // ----- (D) 점수/등급 -----
  float score = 100.0f * (g_rms - SCORE_S0) / (SCORE_S100 - SCORE_S0);
  if (score < 0) score = 0;
  if (score > 100) score = 100;

  const char* grade = "안정";
  if      (g_rms >= 0.30f) grade = "나쁨";
  else if (g_rms >= 0.10f) grade = "주의";
  else if (g_rms >= 0.03f) grade = "양호";

  // ----- (E) 출력 -----
  if (now - lastPrintMs >= PRINT_DT_MS) {
    lastPrintMs = now;

    // 간단 막대(20칸) 시각화
    int bars = (int)(score / 5.0f); // 0~20

    Serial.print(now); Serial.print(F(" | "));

    if (score < 100) Serial.print(' ');
    if (score < 10)  Serial.print(' ');
    Serial.print((int)score); Serial.print(F("    "));

    Serial.print(grade); Serial.print(F("   "));

    for (int i=0;i<20;i++) Serial.print(i<bars ? '#' : '.');
    Serial.print(F("  "));

    // 참고용 수치
    Serial.print(g_rms,3); Serial.print('\t');
    Serial.println(a_rms,3);
  }
}