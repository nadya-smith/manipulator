#include <Wire.h>

// ===== TCA9548A =====
#define TCA9548A_ADDR     0x70    // Адрес мультиплексора (A0=A1=A2=GND)
#define SENSOR_COUNT      7       // Количество датчиков (каналы 0–6)

// ===== AS5600 =====
#define AS5600_ADDR       0x36
#define REG_RAW_ANGLE_H   0x0C
#define REG_ANGLE_H       0x0E
#define REG_STATUS        0x0B
#define REG_AGC           0x1A
#define REG_MAGNITUDE_H   0x1B

// Интервал отправки (мс)
const unsigned long SEND_INTERVAL = 50;   // 20 Гц
unsigned long lastSendTime = 0;

// Маска обнаруженных датчиков
uint8_t sensorMask = 0;

// ===== Выбор канала мультиплексора =====
bool tcaSelect(uint8_t channel) {
  if (channel >= 8) return false;
  Wire.beginTransmission(TCA9548A_ADDR);
  Wire.write(1 << channel);
  return (Wire.endTransmission() == 0);
}

// Отключить все каналы
void tcaDisable() {
  Wire.beginTransmission(TCA9548A_ADDR);
  Wire.write(0);
  Wire.endTransmission();
}

// ===== Чтение регистров AS5600 =====
uint8_t readReg8(uint8_t reg) {
  Wire.beginTransmission(AS5600_ADDR);
  Wire.write(reg);
  Wire.endTransmission(false);
  Wire.requestFrom((uint8_t)AS5600_ADDR, (uint8_t)1);
  return Wire.available() ? Wire.read() : 0;
}

uint16_t readReg16(uint8_t regH) {
  Wire.beginTransmission(AS5600_ADDR);
  Wire.write(regH);
  Wire.endTransmission(false);
  Wire.requestFrom((uint8_t)AS5600_ADDR, (uint8_t)2);
  uint16_t val = 0;
  if (Wire.available() >= 2) {
    val  = (uint16_t)Wire.read() << 8;
    val |= Wire.read();
  }
  return val;
}

// ===== Получение данных с AS5600 =====
uint16_t getRawAngle()    { return readReg16(REG_RAW_ANGLE_H) & 0x0FFF; }
uint8_t  getAGC()         { return readReg8(REG_AGC); }
uint16_t getMagnitude()   { return readReg16(REG_MAGNITUDE_H) & 0x0FFF; }

const char* getMagnetStatus() {
  uint8_t s = readReg8(REG_STATUS) & 0x38;
  if (s & 0x20) return "OK";
  if (s & 0x10) return "WEAK";
  if (s & 0x08) return "STRONG";
  return "NONE";
}

// Проверка наличия AS5600 на текущем канале
bool isAS5600Connected() {
  Wire.beginTransmission(AS5600_ADDR);
  return (Wire.endTransmission() == 0);
}

// Проверка наличия мультиплексора
bool isTCAConnected() {
  Wire.beginTransmission(TCA9548A_ADDR);
  return (Wire.endTransmission() == 0);
}

// ===== SETUP =====
void setup() {
  Serial.begin(115200);
  Wire.begin();
  Wire.setClock(400000);
  delay(500);

  Serial.println("# TCA9548A + 7x AS5600 System Ready");

  // --- Проверка мультиплексора ---
  if (!isTCAConnected()) {
    Serial.println("# ERROR: TCA9548A not found at 0x70!");
    while (1) delay(1000);
  }
  Serial.println("# TCA9548A detected at 0x70");

  // --- Сканирование датчиков на каждом канале ---
  sensorMask = 0;
  for (uint8_t ch = 0; ch < SENSOR_COUNT; ch++) {
    tcaSelect(ch);
    delay(5);  // Небольшая пауза после переключения

    if (isAS5600Connected()) {
      sensorMask |= (1 << ch);
      Serial.print("# CH");
      Serial.print(ch);
      Serial.print(": AS5600 found (magnet: ");
      Serial.print(getMagnetStatus());
      Serial.print(", AGC: ");
      Serial.print(getAGC());
      Serial.println(")");
    } else {
      Serial.print("# CH");
      Serial.print(ch);
      Serial.println(": no AS5600");
    }
  }

  tcaDisable();

  // Итог
  uint8_t count = 0;
  for (uint8_t i = 0; i < SENSOR_COUNT; i++) {
    if (sensorMask & (1 << i)) count++;
  }

  Serial.print("# Total sensors found: ");
  Serial.print(count);
  Serial.print("/");
  Serial.println(SENSOR_COUNT);

  if (count == 0) {
    Serial.println("# WARNING: No sensors detected! Check wiring.");
  }

  Serial.println("# ──────────────────────────────────────");
  Serial.println("# Format: AS5600:ch,deg,raw,agc,mag,status");
  Serial.println("# ──────────────────────────────────────");
}

// ===== LOOP =====
void loop() {
  unsigned long now = millis();

  if (now - lastSendTime >= SEND_INTERVAL) {
    lastSendTime = now;

    // --- Опрос всех обнаруженных датчиков ---
    for (uint8_t ch = 0; ch < SENSOR_COUNT; ch++) {

      // Пропуск каналов без датчиков (опционально — можно убрать
      // для горячего подключения, тогда будет пере-сканирование)
      if (!(sensorMask & (1 << ch))) continue;

      if (!tcaSelect(ch)) {
        // Мультиплексор не отвечает
        Serial.print("AS5600:");
        Serial.print(ch);
        Serial.println(",ERR_MUX");
        continue;
      }

      // Проверяем, что датчик ещё на месте
      if (!isAS5600Connected()) {
        Serial.print("AS5600:");
        Serial.print(ch);
        Serial.println(",ERR_NODEV");
        continue;
      }

      uint16_t    raw = getRawAngle();
      float       deg = (float)raw * 360.0 / 4096.0;
      uint8_t     agc = getAGC();
      uint16_t    mag = getMagnitude();
      const char* st  = getMagnetStatus();

      // ═══ Формат: AS5600:канал,градусы,raw,agc,magnitude,статус ═══
      Serial.print("AS5600:");
      Serial.print(ch);
      Serial.print(",");
      Serial.print(deg, 2);
      Serial.print(",");
      Serial.print(raw);
      Serial.print(",");
      Serial.print(agc);
      Serial.print(",");
      Serial.print(mag);
      Serial.print(",");
      Serial.println(st);
    }

    tcaDisable();   // Отключаем все каналы между циклами
  }

  // ===== Обработка входящих команд из Python =====
  if (Serial.available()) {
    String cmd = Serial.readStringUntil('\n');
    cmd.trim();

    if (cmd == "PING") {
      Serial.println("PONG");
    }
    else if (cmd == "SCAN") {
      // Повторное сканирование датчиков
      sensorMask = 0;
      for (uint8_t ch = 0; ch < SENSOR_COUNT; ch++) {
        tcaSelect(ch);
        delay(5);
        if (isAS5600Connected()) {
          sensorMask |= (1 << ch);
          Serial.print("# SCAN CH");
          Serial.print(ch);
          Serial.println(": FOUND");
        } else {
          Serial.print("# SCAN CH");
          Serial.print(ch);
          Serial.println(": EMPTY");
        }
      }
      tcaDisable();

      uint8_t count = 0;
      for (uint8_t i = 0; i < SENSOR_COUNT; i++) {
        if (sensorMask & (1 << i)) count++;
      }
      Serial.print("# SCAN TOTAL: ");
      Serial.println(count);
    }
    else if (cmd == "STATUS") {
      // Выдать маску подключённых датчиков
      Serial.print("# MASK: 0b");
      for (int8_t i = SENSOR_COUNT - 1; i >= 0; i--) {
        Serial.print((sensorMask >> i) & 1);
      }
      Serial.println();
    }
  }
}