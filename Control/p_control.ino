// Double P controller, reading camera position over Serial1 from RPi,
// controlling X and Y steppers with P controllers to maintain desired position.
// Logs data to SD card.

#include <Arduino.h>
#include <Wire.h>
#include <SPI.h>
#include <SD.h>
#include "I2Cdev.h"
#include "MPU6050_6Axis_MotionApps20.h"

// ===================== CONFIG =====================

// Logging
const float LOG_HZ       = 60.0f;   // [Hz]
const float T_LOG_TOTAL  = 30.0f;    // [s]
const float PRE_ROLL_S   = 1.0f;     // [s]

// Camera calibration (mm from Pi are too small by factor 1/1.17)
const float CAM_MM_SCALE = 1.17f;    // real_mm = cam_mm * 1.17  // 
const float CAM_MM_SCALE_Y = 1.19f;  
// Belt axis sign (WORLD Y): +1 or -1
const int   BELT_SIGN    = +1;

// ===== P CONTROLLERS ON CAMERA X/Y POSITION (mm) =====
const float REF_X_MM     = 0.0f;   // target camera X position [mm]
const float REF_Y_MM     = 50.0f;    // target camera Y position [mm]
const float KP_CAM_X     = 40.0f;    // [steps/s per mm error]
const float KP_CAM_Y     = 80.0f;    // [steps/s per mm error]

// ===================== SD LOGGING =====================

const int   SD_CS_PIN      = BUILTIN_SDCARD;
const char *LOG_FILENAME   = "DoublePController5.csv";
File        logFile;

// ===================== AS5600 (absolute encoders) =====================
//
// X encoder on I2C0 (Wire), Y encoder on I2C1 (Wire1), same address.

#define AS5600_ADDR   0x36
#define AS5600_RAW_H  0x0C
#define AS5600_RAW_L  0x0D

float readEncoderXDeg() {
  Wire.beginTransmission(AS5600_ADDR);
  Wire.write(AS5600_RAW_H);
  if (Wire.endTransmission(false) != 0) return NAN;

  Wire.requestFrom((uint8_t)AS5600_ADDR, (uint8_t)2);
  if (Wire.available() < 2) return NAN;

  uint16_t hi  = Wire.read();
  uint16_t lo  = Wire.read();
  uint16_t raw = ((hi << 8) | lo) & 0x0FFF;     // 12-bit

  return raw * 360.0f / 4096.0f;
}

float readEncoderYDeg() {
  Wire1.beginTransmission(AS5600_ADDR);
  Wire1.write(AS5600_RAW_H);
  if (Wire1.endTransmission(false) != 0) return NAN;

  Wire1.requestFrom((uint8_t)AS5600_ADDR, (uint8_t)2);
  if (Wire1.available() < 2) return NAN;

  uint16_t hi  = Wire1.read();
  uint16_t lo  = Wire1.read();
  uint16_t raw = ((hi << 8) | lo) & 0x0FFF;     // 12-bit

  return raw * 360.0f / 4096.0f;
}

// ===================== X STEPPER (belt, TB6600 PUL+ / DIR+) =====================

const uint8_t X_STEP_PIN = 9;  // Teensy 9 -> TB6600 PUL+
const uint8_t X_DIR_PIN  = 8;  // Teensy 8 -> TB6600 DIR+

// ===================== Y STEPPERS (two motors, separate STEP) =====================
//
// Motor 1: DIR1 = 10, STEP1 = 11
// Motor 2: DIR2 = 6,  STEP2 = 7

const uint8_t Y_STEP1_PIN = 11;
const uint8_t Y_DIR1_PIN  = 10;
const uint8_t Y_STEP2_PIN = 7;
const uint8_t Y_DIR2_PIN  = 6;

// ===================== STEP ENGINE CONFIG =====================

const uint32_t ISR_TICK_HZ   = 30000UL; // step engine tick rate
const uint16_t MAX_SPS_X     = 2500;    // max safe steps/s for X
const uint16_t MAX_SPS_Y     = 2500;    // max safe steps/s for Y
const uint16_t MIN_SPS       = 350;     // min usable steps/s (both axes)

// ===================== STEPPER AXIS STRUCTS =====================

struct StepperAxisX {
  uint8_t stepPin = 255;
  uint8_t dirPin  = 255;

  volatile uint16_t halfPeriodTicks = 0;
  volatile uint16_t tickCounter     = 0;
  volatile int8_t  dirSign          = +1;
  volatile bool    stepState        = false;
  volatile long    stepCount        = 0;
  volatile float   cmdSps           = 0.0f;

  void attach(uint8_t stepPin_, uint8_t dirPin_) {
    stepPin = stepPin_;
    dirPin  = dirPin_;
    pinMode(stepPin, OUTPUT);
    pinMode(dirPin, OUTPUT);
    digitalWriteFast(stepPin, LOW);
    digitalWriteFast(dirPin, LOW);
  }

  inline void setDirection(int8_t sign) {
    dirSign = (sign >= 0) ? +1 : -1;
    digitalWriteFast(dirPin, (dirSign > 0 ? HIGH : LOW));
  }

  void setSpeed(float sps_mag) {
    if (stepPin == 255) return;

    float mag = sps_mag;

    // Enforce minimum and maximum speed (for non-zero motion)
    if (mag < MIN_SPS) mag = MIN_SPS;
    if (mag > MAX_SPS_X) mag = MAX_SPS_X;

    double hpt_f    = (double)ISR_TICK_HZ / (2.0 * mag);
    uint16_t newHpt = (uint16_t)max(1u, min(65535u, (uint32_t)(hpt_f + 0.5)));

    noInterrupts();
    halfPeriodTicks = newHpt;
    tickCounter     = newHpt;
    cmdSps          = mag;
    interrupts();
  }

  void stop() {
    noInterrupts();
    halfPeriodTicks = 0;
    tickCounter     = 0;
    cmdSps          = 0.0f;
    interrupts();
  }

  inline void isrTick() {
    uint16_t hpt = halfPeriodTicks;
    if (hpt == 0 || stepPin == 255) return;

    uint16_t c = tickCounter;
    if (--c == 0) {
      c = hpt;
      stepState = !stepState;
      if (stepState) {
        digitalWriteFast(stepPin, HIGH);
        stepCount += dirSign;
      } else {
        digitalWriteFast(stepPin, LOW);
      }
    }
    tickCounter = c;
  }
};

struct StepperAxisY {
  uint8_t stepPin1 = 255;
  uint8_t dirPin1  = 255;
  uint8_t stepPin2 = 255;
  uint8_t dirPin2  = 255;
  bool    invertDir2 = true;   // set true if second motor is mirrored

  volatile uint16_t halfPeriodTicks = 0;
  volatile uint16_t tickCounter     = 0;
  volatile int8_t  dirSign          = +1;
  volatile bool    stepState        = false;
  volatile long    stepCount        = 0;
  volatile float   cmdSps           = 0.0f;

  void attach(uint8_t stepPin1_, uint8_t dirPin1_,
              uint8_t stepPin2_, uint8_t dirPin2_,
              bool invert2 = false) {
    stepPin1   = stepPin1_;
    dirPin1    = dirPin1_;
    stepPin2   = stepPin2_;
    dirPin2    = dirPin2_;
    invertDir2 = invert2;

    pinMode(stepPin1, OUTPUT);
    pinMode(dirPin1,  OUTPUT);
    pinMode(stepPin2, OUTPUT);
    pinMode(dirPin2,  OUTPUT);

    digitalWriteFast(stepPin1, LOW);
    digitalWriteFast(dirPin1,  LOW);
    digitalWriteFast(stepPin2, LOW);
    digitalWriteFast(dirPin2,  LOW);
  }

  inline void setDirection(int8_t sign) {
    dirSign = (sign >= 0) ? +1 : -1;

    bool d1 = (dirSign > 0);
    bool d2 = invertDir2 ? !d1 : d1;

    digitalWriteFast(dirPin1, d1 ? HIGH : LOW);
    digitalWriteFast(dirPin2, d2 ? HIGH : LOW);
  }

  void setSpeed(float sps_mag) {
    if (stepPin1 == 255 || stepPin2 == 255) return;

    float mag = sps_mag;

    if (mag < MIN_SPS) mag = MIN_SPS;
    if (mag > MAX_SPS_Y) mag = MAX_SPS_Y;

    double hpt_f    = (double)ISR_TICK_HZ / (2.0 * mag);
    uint16_t newHpt = (uint16_t)max(1u, min(65535u, (uint32_t)(hpt_f + 0.5)));

    noInterrupts();
    halfPeriodTicks = newHpt;
    tickCounter     = newHpt;
    cmdSps          = mag;
    interrupts();
  }

  void stop() {
    noInterrupts();
    halfPeriodTicks = 0;
    tickCounter     = 0;
    cmdSps          = 0.0f;
    interrupts();
  }

  inline void isrTick() {
    uint16_t hpt = halfPeriodTicks;
    if (hpt == 0 || stepPin1 == 255 || stepPin2 == 255) return;

    uint16_t c = tickCounter;
    if (--c == 0) {
      c = hpt;
      stepState = !stepState;
      if (stepState) {
        digitalWriteFast(stepPin1, HIGH);
        digitalWriteFast(stepPin2, HIGH);
        stepCount += dirSign;
      } else {
        digitalWriteFast(stepPin1, LOW);
        digitalWriteFast(stepPin2, LOW);
      }
    }
    tickCounter = c;
  }
};

StepperAxisX axisX;
StepperAxisY axisY;

// ===================== TIMERS & GLOBAL STATE =====================

IntervalTimer stepTimerX;
IntervalTimer stepTimerY;

unsigned long g_t0_us   = 0;
bool          g_logDone = false;

// ===================== MPU6050 + DMP =====================

MPU6050  mpu;
uint8_t  devStatus;
bool     dmpReady   = false;
uint16_t packetSize = 0;
uint16_t fifoCount  = 0;
uint8_t  fifoBuffer[64];

Quaternion   q;
VectorInt16  aa, aaReal, aaWorld;
VectorFloat  gravity;

float        v_belt = 0.0f;
elapsedMicros dt_us;

// IMU-derived quantities (kept for observer/logging)
float axW_last    = 0.0f;
float ayW_last    = 0.0f;
float azW_last    = 0.0f;
float a_belt_last = 0.0f;

// accel LSB -> m/s^2
inline void accelLSBtoSI(const VectorInt16 &v, float &x, float &y, float &z) {
  const float g_per_lsb = 1.0f / 16384.0f;
  const float g_to_ms2  = 9.80665f;
  x = v.x * g_per_lsb * g_to_ms2;
  y = v.y * g_per_lsb * g_to_ms2;
  z = v.z * g_per_lsb * g_to_ms2;
}

// ===================== CAMERA DATA (RPi over Serial1) =====================
//
// Pi sends lines: t_teensy,x_mm,y_mm,dx_mm,dy_mm,vx_mm,vy_mm\n
// Teensy sends time tags: "T,<t_s>\n"

float cam_t_s   = NAN;
float cam_x_mm  = NAN;
float cam_y_mm  = NAN;
float cam_dx_mm = NAN;
float cam_dy_mm = NAN;
float cam_vx_mm = NAN;
float cam_vy_mm = NAN;

void handlePiCameraData() {
  while (Serial1.available() > 0) {
    static char    buf[96];
    static uint8_t idx = 0;
    char           c   = Serial1.read();

    if (c == '\n' || c == '\r') {
      if (idx == 0) continue;
      buf[idx] = '\0';
      idx      = 0;

      float t, x, y, dx, dy, vx, vy;
      int   n = sscanf(buf, "%f,%f,%f,%f,%f,%f,%f",
                       &t, &x, &y, &dx, &dy, &vx, &vy);
      if (n == 7) {
        cam_t_s   = t;
        cam_x_mm  = x;
        cam_y_mm  = y;
        cam_dx_mm = dx;
        cam_dy_mm = dy;
        cam_vx_mm = vx;
        cam_vy_mm = vy;
      }
    } else {
      if (idx < sizeof(buf) - 1) {
        buf[idx++] = c;
      }
    }
  }
}

// Step pulse ISRs
void stepTimerXISR() { axisX.isrTick(); }
void stepTimerYISR() { axisY.isrTick(); }

// ===================== SETUP =====================

void setup() {
  Serial.begin(115200);    // to PC
  Serial1.begin(115200);   // to RPi

  while (!Serial && millis() < 4000) {
    // optional wait
  }

  // --- SD init ---
  if (!SD.begin(SD_CS_PIN)) {
    Serial.println("SD init FAILED");
    while (1) {}
  }

  if (SD.exists(LOG_FILENAME)) {
    SD.remove(LOG_FILENAME);
  }

  logFile = SD.open(LOG_FILENAME, FILE_WRITE);
  if (!logFile) {
    Serial.println("Failed to open log file");
    while (1) {}
  }

  // CSV header
  logFile.println(
    "t_s,"
    "enc_x_deg,"
    "acc_x_deg,"
    "enc_y_deg,"
    "acc_y_deg,"
    "axW,"
    "ayW,"
    "azW,"
    "a_belt,"
    "v_belt,"
    "cmd_sps_x,"
    "cmd_dir_x,"
    "cmd_vel_x,"
    "steps_x,"
    "cmd_sps_y,"
    "cmd_dir_y,"
    "cmd_vel_y,"
    "steps_y,"
    "cam_t_s,"
    "cam_x_mm,"
    "cam_y_mm,"
    "cam_dx_mm,"
    "cam_dy_mm,"
    "cam_vx_mm,"
    "cam_vy_mm"
  );
  logFile.flush();

  Wire.begin();   // I2C0 (X AS5600)
  Wire1.begin();  // I2C1 (Y AS5600)

  // --- IMU + DMP init ---
  mpu.initialize();
  if (!mpu.testConnection()) {
    Serial.println("MPU6050 connection FAILED");
    while (1) {}
  }

  Serial.println("Initializing DMP...");
  devStatus = mpu.dmpInitialize();
  if (devStatus != 0) {
    Serial.print("DMP init failed (code ");
    Serial.print(devStatus);
    Serial.println(")");
    while (1) {}
  }

  Serial.println("Calibrating IMU... keep still.");
  mpu.CalibrateAccel(6);
  mpu.CalibrateGyro(6);

  mpu.setDMPEnabled(true);
  mpu.resetFIFO();
  packetSize = mpu.dmpGetFIFOPacketSize();
  dmpReady   = true;
  dt_us      = 0;

  // --- Steppers ---
  axisX.attach(X_STEP_PIN, X_DIR_PIN);
  // Set last bool to true if second motor is mechanically mirrored
  axisY.attach(Y_STEP1_PIN, Y_DIR1_PIN, Y_STEP2_PIN, Y_DIR2_PIN, false);

  delay(2000);  // let everything settle

  g_t0_us   = micros();
  g_logDone = false;
  v_belt    = 0.0f;

  double stepPeriod_us = 1000000.0 / (double)ISR_TICK_HZ;
  stepTimerX.begin(stepTimerXISR, stepPeriod_us);
  stepTimerY.begin(stepTimerYISR, stepPeriod_us);

  Serial.println("System ready: 2D P control on cam_x_mm, cam_y_mm, logging to SD (DoublePController.csv).");
}

// ===================== LOOP =====================

void loop() {
  float t_s = (micros() - g_t0_us) * 1e-6f;

  // Stop after T_LOG_TOTAL
  if (!g_logDone && t_s > T_LOG_TOTAL) {
    axisX.stop();
    axisY.stop();
    g_logDone = true;

    if (logFile) {
      logFile.flush();
      logFile.close();
    }
    Serial.println("# Logging done, motors stopped, file closed.");
  }
  if (g_logDone) {
    handlePiCameraData();
    delay(10);
    return;
  }

  // Send time tag to Pi
  Serial1.print("T,");
  Serial1.println(t_s, 3);

  // Update camera data
  handlePiCameraData();

  // Snapshot step positions
  long step_pos_x, step_pos_y;
  noInterrupts();
  step_pos_x = axisX.stepCount;
  step_pos_y = axisY.stepCount;
  interrupts();

  // ===================== P CONTROLLER X (cam_x_mm) =====================
  float cmd_sps_x = 0.0f;
  int8_t cmd_dir_x = +1;
  float  cmd_vel_x = 0.0f;

  if (t_s >= PRE_ROLL_S && !isnan(cam_x_mm)) {
    float cam_x_real_mm = cam_x_mm * CAM_MM_SCALE;       
    float err_x_mm      = REF_X_MM - cam_x_real_mm;       

    float u = KP_CAM_X * err_x_mm;

    if (u > 0.0f) {
      cmd_dir_x = +1;
      cmd_sps_x = u;
    } else if (u < 0.0f) {
      cmd_dir_x = -1;
      cmd_sps_x = -u;
    } else {
      cmd_dir_x = +1;
      cmd_sps_x = 0.0f;
    }

    if (fabsf(err_x_mm) < 3.0f) { // 3 mm deadband
      cmd_sps_x = 0.0f;
    }

    if (cmd_sps_x <= 0.0f) {
      axisX.stop();
      cmd_sps_x = 0.0f;
    } else {
      if (cmd_sps_x > MAX_SPS_X) cmd_sps_x = MAX_SPS_X;
      axisX.setDirection(cmd_dir_x);
      axisX.setSpeed(cmd_sps_x);     // enforces MIN/MAX
    }

    cmd_vel_x = cmd_sps_x * (float)cmd_dir_x;
  } else {
    axisX.stop();
    cmd_sps_x = 0.0f;
    cmd_dir_x = +1;
    cmd_vel_x = 0.0f;
  }

  // ===================== P CONTROLLER Y (cam_y_mm) =====================
  float cmd_sps_y = 0.0f;
  int8_t cmd_dir_y = +1;
  float  cmd_vel_y = 0.0f;

  if (t_s >= PRE_ROLL_S && !isnan(cam_y_mm)) {
    float cam_y_real_mm = cam_y_mm * CAM_MM_SCALE_Y;        
    // flipped sign as we discussed earlier
    float err_y_mm      = cam_y_real_mm - REF_Y_MM;      

    float u = KP_CAM_Y * err_y_mm;

    if (u > 0.0f) {
      cmd_dir_y = +1;
      cmd_sps_y = u;
    } else if (u < 0.0f) {
      cmd_dir_y = -1;
      cmd_sps_y = -u;
    } else {
      cmd_dir_y = +1;
      cmd_sps_y = 0.0f;
    }

    if (fabsf(err_y_mm) < 3.0f) { // 3 mm deadband
      cmd_sps_y = 0.0f;
    }

    if (cmd_sps_y <= 0.0f) {
      axisY.stop();
      cmd_sps_y = 0.0f;
    } else {
      if (cmd_sps_y > MAX_SPS_Y) cmd_sps_y = MAX_SPS_Y;
      axisY.setDirection(cmd_dir_y);
      axisY.setSpeed(cmd_sps_y);     // enforces MIN/MAX
    }

    cmd_vel_y = cmd_sps_y * (float)cmd_dir_y;
  } else {
    axisY.stop();
    cmd_sps_y = 0.0f;
    cmd_dir_y = +1;
    cmd_vel_y = 0.0f;
  }

  // ===================== IMU / DMP (kept for observer) =====================
  if (dmpReady) {
    fifoCount = mpu.getFIFOCount();
    if (fifoCount == 1024) {
      mpu.resetFIFO();
      Serial.println("# FIFO overflow, resetting.");
    } else if (fifoCount >= packetSize) {
      while (fifoCount >= packetSize) {
        mpu.getFIFOBytes(fifoBuffer, packetSize);
        fifoCount -= packetSize;
      }

      mpu.dmpGetQuaternion(&q, fifoBuffer);
      mpu.dmpGetAccel(&aa, fifoBuffer);
      mpu.dmpGetGravity(&gravity, &q);
      mpu.dmpGetLinearAccel(&aaReal, &aa, &gravity);
      mpu.dmpGetLinearAccelInWorld(&aaWorld, &aaReal, &q);

      float axW, ayW, azW;
      accelLSBtoSI(aaWorld, axW, ayW, azW);

      float a_rod = azW;

      float a_belt = BELT_SIGN * ayW;

      float dt = dt_us / 1e6f;
      dt_us = 0;
      v_belt += a_belt * dt;

      axW_last    = axW;
      ayW_last    = ayW;
      azW_last    = azW;
      a_belt_last = a_belt;
    }
  }

  // ===================== Encoders X/Y: accumulate angle =====================
  static float acc_as5600_x_deg  = 0.0f;
  static float prev_as5600_x_deg = NAN;
  static float acc_as5600_y_deg  = 0.0f;
  static float prev_as5600_y_deg = NAN;

  float enc_x_deg = readEncoderXDeg();
  if (!isnan(enc_x_deg)) {
    if (isnan(prev_as5600_x_deg)) {
      prev_as5600_x_deg = enc_x_deg;
      acc_as5600_x_deg  = 0.0f;
    } else {
      float delta = enc_x_deg - prev_as5600_x_deg;
      if (delta > 180.0f)  delta -= 360.0f;
      if (delta < -180.0f) delta += 360.0f;
      acc_as5600_x_deg += delta;
      prev_as5600_x_deg = enc_x_deg;
    }
  }

  float enc_y_deg = readEncoderYDeg();
  if (!isnan(enc_y_deg)) {
    if (isnan(prev_as5600_y_deg)) {
      prev_as5600_y_deg = enc_y_deg;
      acc_as5600_y_deg  = 0.0f;
    } else {
      float delta = enc_y_deg - prev_as5600_y_deg;
      if (delta > 180.0f)  delta -= 360.0f;
      if (delta < -180.0f) delta += 360.0f;
      acc_as5600_y_deg += delta;
      prev_as5600_y_deg = enc_y_deg;
    }
  }

  // ===== Correct camera values for logging (optional but nice) =====  
  float cam_x_corr   = isnan(cam_x_mm)  ? NAN : cam_x_mm  * CAM_MM_SCALE;
  float cam_y_corr   = isnan(cam_y_mm)  ? NAN : cam_y_mm  * CAM_MM_SCALE_Y;
  float cam_dx_corr  = isnan(cam_dx_mm) ? NAN : cam_dx_mm * CAM_MM_SCALE;
  float cam_dy_corr  = isnan(cam_dy_mm) ? NAN : cam_dy_mm * CAM_MM_SCALE_Y;
  float cam_vx_corr  = isnan(cam_vx_mm) ? NAN : cam_vx_mm * CAM_MM_SCALE;
  float cam_vy_corr  = isnan(cam_vy_mm) ? NAN : cam_vy_mm * CAM_MM_SCALE_Y;

  // ===================== LOGGING TO SD =====================

  static float lastLog_t = 0.0f;
  if (t_s - lastLog_t < 1.0f / LOG_HZ) {
    return;
  }
  lastLog_t = t_s;

  if (logFile && !g_logDone) {
    logFile.print(t_s, 5);            logFile.print(',');
    logFile.print(enc_x_deg, 3);      logFile.print(',');
    logFile.print(acc_as5600_x_deg, 3); logFile.print(',');
    logFile.print(enc_y_deg, 3);      logFile.print(',');
    logFile.print(acc_as5600_y_deg, 3); logFile.print(',');
    logFile.print(axW_last, 3);       logFile.print(',');
    logFile.print(ayW_last, 3);       logFile.print(',');
    logFile.print(azW_last, 3);       logFile.print(',');
    logFile.print(a_belt_last, 3);    logFile.print(',');
    logFile.print(v_belt, 3);         logFile.print(',');
    logFile.print(cmd_sps_x, 1);      logFile.print(',');
    logFile.print((int)cmd_dir_x);    logFile.print(',');
    logFile.print(cmd_vel_x, 1);      logFile.print(',');
    logFile.print(step_pos_x);        logFile.print(',');
    logFile.print(cmd_sps_y, 1);      logFile.print(',');
    logFile.print((int)cmd_dir_y);    logFile.print(',');
    logFile.print(cmd_vel_y, 1);      logFile.print(',');
    logFile.print(step_pos_y);        logFile.print(',');
    logFile.print(cam_t_s, 3);        logFile.print(',');
    logFile.print(cam_x_corr, 3);     logFile.print(',');   
    logFile.print(cam_y_corr, 3);     logFile.print(',');   
    logFile.print(cam_dx_corr, 3);    logFile.print(',');   
    logFile.print(cam_dy_corr, 3);    logFile.print(',');   
    logFile.print(cam_vx_corr, 3);    logFile.print(',');   
    logFile.println(cam_vy_corr, 3);                      

    // Flush occasionally to avoid data loss but not every sample
    static uint32_t logCount = 0;
    if (++logCount % 20 == 0) { // ~2x per second at 100 Hz
      logFile.flush();
    }
  }
}
