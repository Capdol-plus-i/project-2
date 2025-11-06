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

// 시각화 스케일 (지금은 안 써도 됨)
const float GYRO_FULL_SCALE     = 1.0f;
const float ARMS_FULL_SCALE_G   = 1.0f;

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

// IIR RMS (제곱 평균의 지수평활)
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
  Serial.begin(9600);
  delay(300);

  Wire.begin();
#if defined(TWBR) || ARDUINO >= 100
  Wire.setClock(400000); // I2C 400 kHz
#endif

  if (!beginAuto()) {
    Serial.println(F("[ERR] MPU6050 not found (0x68/0x69). 배선/I2C 확인."));
    while (1) { delay(500); }
  }

  // 센서 설정
  mpu.setFilterBandwidth(MPU6050_BAND_44_HZ);
  mpu.setAccelerometerRange(MPU6050_RANGE_4_G);
  mpu.setGyroRange(MPU6050_RANGE_500_DEG);
  delay(150);

  // 필터 계수 계산
  const float dt = SAMPLE_DT_MS / 1000.0f;
  const float tau_hp = 1.0f / (2.0f * M_PI * HP_CUTOFF_HZ);
  const float tau_lp = 1.0f / (2.0f * M_PI * LP_CUTOFF_HZ);
  hp_a = tau_hp / (tau_hp + dt);   // HPF: y[n]=a*(y[n-1]+x[n]-x[n-1])
  lp_b = dt / (tau_lp + dt);       // LPF: y[n]=y[n-1]+b*(x[n]-y[n-1])

  // IIR RMS 시간상수
  const float tau_rms = 0.1f;
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

  // 이제부터는 주기 출력이 숫자 3개만 나올 거라고 알려주자
  Serial.println(F("[READY] format: millis,g_rms_rad_s,a_rms_g"));
}

void loop() {
  unsigned long now = millis();
  if (now - lastMs < SAMPLE_DT_MS) return;
  lastMs += SAMPLE_DT_MS;

  sensors_event_t a, g, t;
  mpu.getEvent(&a, &g, &t);

  // ----- (A) accel (g) offset 제거 -----
  float ax = a.acceleration.x / G_CONST - offAX;
  float ay = a.acceleration.y / G_CONST - offAY;
  float az = a.acceleration.z / G_CONST - offAZ;

  // 밴드패스 8~40Hz
  ahp_x = hp_a * (ahp_x + ax - prev_ax);
  ahp_y = hp_a * (ahp_y + ay - prev_ay);
  ahp_z = hp_a * (ahp_z + az - prev_az);
  prev_ax = ax; prev_ay = ay; prev_az = az;

  alp_x = alp_x + lp_b * (ahp_x - alp_x);
  alp_y = alp_y + lp_b * (ahp_y - alp_y);
  alp_z = alp_z + lp_b * (ahp_z - alp_z);

  float amag = sqrtf(alp_x*alp_x + alp_y*alp_y + alp_z*alp_z); // g

  // ----- (B) gyro (rad/s) offset 제거 -----
  float gy = g.gyro.y - offGY;
  float gz = g.gyro.z - offGZ;

  ghp_y = hp_a * (ghp_y + gy - prev_gy);
  ghp_z = hp_a * (ghp_z + gz - prev_gz);
  prev_gy = gy; prev_gz = gz;

  glp_y = glp_y + lp_b * (ghp_y - glp_y);
  glp_z = glp_z + lp_b * (ghp_z - glp_z);

  float gmag = sqrtf(glp_y*glp_y + glp_z*glp_z); // rad/s

  // ----- (C) IIR RMS 업데이트 -----
  a_rms2_iir = a_rms2_iir + rms_beta * (amag*amag - a_rms2_iir);
  g_rms2_iir = g_rms2_iir + rms_beta * (gmag*gmag - g_rms2_iir);
  float a_rms = sqrtf(a_rms2_iir);
  float g_rms = sqrtf(g_rms2_iir);

  // ----- (D) 출력: 숫자만 -----
  if (now - lastPrintMs >= PRINT_DT_MS) {
    lastPrintMs = now;

    // PC 파서가 먹기 쉬운 포맷: millis,g_rms,a_rms
    Serial.print(now);
    Serial.print(',');
    Serial.print(g_rms, 3);
    Serial.print(',');
    Serial.println(a_rms, 3);
  }
}