#include <Wire.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include <math.h>

Adafruit_MPU6050 mpu;

// ===== 사용자 설정 =====
const unsigned long CALIB_MS      = 3000;    // 캘리브레이션 시간(정지 권장)
const unsigned long SAMPLE_DT_MS  = 10;      // 센서 샘플 주기(~100 Hz)
const unsigned long PRINT_DT_MS   = 100;     // 모니터 출력 주기(10 Hz)
const float         ALPHA         = 0.20f;   // EMA 계수
const float         G_CONST       = 9.80665f;// 중력가속도 (m/s^2)

// ===== 내부 상태 =====
float offX = 0, offY = 0, offZ = 0; // 오프셋(g)
float fx = 0,  fy = 0,  fz = 0;     // 필터된 값(g)
unsigned long lastMs = 0, lastPrintMs = 0;

bool beginAuto() {
  if (mpu.begin(0x68)) return true; // AD0=GND(기본)
  if (mpu.begin(0x69)) return true; // AD0=VCC
  return false;
}

void calibrateOffsets();

void setup() {
  Serial.begin(115200);      // ⬅ Serial Monitor 대역 확보
  delay(500);                // 보드 안정화

  Wire.begin();
#if defined(TWBR) || ARDUINO >= 100
  Wire.setClock(400000);     // I2C 400kHz (가능한 보드에서)
#endif

  if (!beginAuto()) {
    Serial.println(F("[ERR] MPU6050 not found at 0x68/0x69. 배선/I2C 주소 확인."));
    while (1) { delay(500); }
  }

  // 센서 설정
  mpu.setAccelerometerRange(MPU6050_RANGE_4_G);   // 2/4/8/16G
  mpu.setGyroRange(MPU6050_RANGE_500_DEG);
  mpu.setFilterBandwidth(MPU6050_BAND_21_HZ);
  delay(200);

  // 캘리브레이션
  Serial.print(F("[CAL] "));
  Serial.print(CALIB_MS);
  Serial.println(F(" ms 동안 정지해 주세요..."));
  calibrateOffsets();
  Serial.print(F("[CAL] offsets(g): "));
  Serial.print(offX,3); Serial.print(F(", "));
  Serial.print(offY,3); Serial.print(F(", "));
  Serial.println(offZ,3);

  Serial.println(F("[READY] Serial Monitor 115200 baud. 값은 g 단위."));
  Serial.println(F("-----------------------------------------------"));
  Serial.println(F("time(ms) | ax(g)\tay(g)\taz(g)\t|g|(g)"));
  Serial.println(F("-----------------------------------------------"));
}

void loop() {
  unsigned long now = millis();
  if (now - lastMs < SAMPLE_DT_MS) return;
  lastMs += SAMPLE_DT_MS;  // 타임슬롯 고정: 드리프트 감소

  sensors_event_t a, g, t;
  mpu.getEvent(&a, &g, &t);

  // m/s^2 → g
  float ax = a.acceleration.x / G_CONST;
  float ay = a.acceleration.y / G_CONST;
  float az = a.acceleration.z / G_CONST;

  // 오프셋 보정(정지 시 x≈0g, y≈0g, z≈+1g)
  ax -= offX; ay -= offY; az -= offZ;

  // EMA 로우패스
  fx = ALPHA * ax + (1.0f - ALPHA) * fx;
  fy = ALPHA * ay + (1.0f - ALPHA) * fy;
  fz = ALPHA * az + (1.0f - ALPHA) * fz;

  // 합성가속도 (정지 시 ≈ 1g)
  float g_total = sqrtf(fx*fx + fy*fy + fz*fz);

  // 사람이 읽기 좋은 10 Hz 출력
  if (now - lastPrintMs >= PRINT_DT_MS) {
    lastPrintMs = now;
    Serial.print(now);      Serial.print(F(" | "));
    Serial.print(fx,3);     Serial.print('\t');
    Serial.print(fy,3);     Serial.print('\t');
    Serial.print(fz,3);     Serial.print('\t');
    Serial.println(g_total,3);
  }
}

void calibrateOffsets() {
  unsigned long start = millis();
  double sx = 0, sy = 0, sz = 0;
  unsigned long n = 0;

  while (millis() - start < CALIB_MS) {
    sensors_event_t a, g, t;
    mpu.getEvent(&a, &g, &t);
    sx += a.acceleration.x / G_CONST;
    sy += a.acceleration.y / G_CONST;
    sz += a.acceleration.z / G_CONST;
    n++;
    delay(5);
  }

  double ax = sx / n;
  double ay = sy / n;
  double az = sz / n;

  offX = ax;            // 정지시 x≈0g
  offY = ay;            // 정지시 y≈0g
  offZ = az - 1.0f;     // 정지시 z≈+1g가 되도록 보정

  // 필터 초기값 시드(초기 튐 감소)
  fx = 0.0f; fy = 0.0f; fz = 1.0f;
}