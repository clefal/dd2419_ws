// Single Pin mode
#include <Arduino.h>
#include <lx16a-servo.h>

#include <array>

constexpr std::size_t const NUM_SERVO = 6;
bool servo_resetted = false;

struct ServoConfig
{
  float init_pos;
  float min_pos;
  float max_pos;
  uint16_t init_time;
};

constexpr std::array<ServoConfig, NUM_SERVO> const config{
    ServoConfig{40.0f, 0.0f, 168.0f, 3000},
    ServoConfig{120, 0.0f, 240.0f, 3000},
    ServoConfig{30, 0.0f, 240.0f, 3000},
    ServoConfig{220, 0.0f, 240.0f, 3000},
    ServoConfig{180, 0.0f, 240.0f, 3000},
    ServoConfig{120, 0.0f, 240.0f, 3000},
};

// Servos and Servo Bus Instantiation
LX16ABus servo_bus;
std::array<LX16AServo, NUM_SERVO> servo{
    LX16AServo{&servo_bus, 1}, // 0-16800 (0-168 degrees)
    LX16AServo{&servo_bus, 2}, // 0-24000 (0-240 degrees)
    LX16AServo{&servo_bus, 3}, // 0-24000 (0-240 degrees)
    LX16AServo{&servo_bus, 4}, // 0-24000 (0-240 degrees)
    LX16AServo{&servo_bus, 5}, // 0-24000 (0-240 degrees)
    LX16AServo{&servo_bus, 6}, // 0-24000 (0-240 degrees)
};

template <class T>
constexpr T clamp(T v, T lo, T hi)
{
  return v < lo ? lo : v > hi ? hi
                              : v;
}

void move(std::size_t idx, float pos, uint16_t time)
{
  pos = clamp(pos, config[idx].min_pos, config[idx].max_pos);
  servo[idx].enable();
  servo[idx].move_time(static_cast<int32_t>(pos * 100.0f), time);
}

void reset()
{
  if (servo_resetted)
  {
    return;
  }

  servo_resetted = true;

  for (std::size_t i{}; NUM_SERVO > i; ++i)
  {
    move(i, config[i].init_pos, config[i].init_time);
  }

  uint16_t max_time = 0;
  for (std::size_t i{}; NUM_SERVO > i; ++i)
  {
    max_time = std::max(max_time, config[i].init_time);
  }

  delay(max_time);

  for (std::size_t i{}; NUM_SERVO > i; ++i)
  {
    servo[i].disable();
  }
}

void move()
{
  servo_resetted = false;

  std::array<float, NUM_SERVO> angle;
  std::array<uint16_t, NUM_SERVO> time;
  if (sizeof(angle) != Serial.readBytes(reinterpret_cast<uint8_t *>(angle.data()), sizeof(angle)) ||
      sizeof(time) != Serial.readBytes(reinterpret_cast<uint8_t *>(time.data()), sizeof(time)))
  {
    return;
  }

  for (std::size_t i{}; servo.size() > i; ++i)
  {
    move(i, angle[i], time[i]);
  }
}

void feedback()
{
  std::array<float, NUM_SERVO> feedback;
  for (std::size_t i{}; servo.size() > i; ++i)
  {
    feedback[i] = servo[i].pos_read() / 100.0f;
  }

  Serial.print("BEGIN FEEDBACK");
  Serial.write(reinterpret_cast<uint8_t const *>(feedback.data()), sizeof(feedback));
  Serial.print("END FEEDBACK");
}

void setup()
{
  Serial.begin(115200);

  // Servo Setup
  // Serial.println("Beginning Servo Example");
  servo_bus.beginOnePinMode(&Serial2, 33);
  servo_bus.debug(false);
  servo_bus.retry = 1;

  // Reset the servo positions
  reset();

  Serial.println("Arm is ready");
}

void loop()
{
  static std::size_t num_missed = 0;

  if (0 >= Serial.available())
  {
    ++num_missed;
    if (10 == num_missed)
    {
      reset();
    }
    else
    {
      delay(80);
    }
    return;
  }

  num_missed = 0;

  int action = Serial.read();

  if (-1 == action)
  {
    reset();
    return;
  }

  uint8_t a = static_cast<uint8_t>(action);
  if (0b001 & a)
  {
    move();
  }
  if (0b010 & a)
  {
    feedback();
  }
  if (0b100 & a)
  {
    reset();
  }

  delay(80);
}