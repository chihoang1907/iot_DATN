# # test servo
# import RPi.GPIO as GPIO
# from adafruit_servokit import ServoKit
# from time import sleep

# CENTER = 95
# MIN = 65
# MAX = 120
# kit = ServoKit(channels=16)
# servo = kit.servo[0]
# # servo.set_pulse_width_range(500, 2500)
# servo.angle = 95
# sleep(1)

# GPIO.cleanup()

import time

# from board import SCL, SDA
import busio

# Import the PCA9685 module. Available in the bundle and here:
#   https://github.com/adafruit/Adafruit_CircuitPython_PCA9685
from adafruit_motor import servo
from adafruit_pca9685 import PCA9685
import RPi.GPIO as GPIO

GPIO.setmode(GPIO.TEGRA_SOC)
# print mode board to tegra_soc
board_to_tegra_soc = {
k: list(GPIO.gpio_pin_data.get_data()[-1]['TEGRA_SOC'].keys())[i] for i, k in enumerate(GPIO.gpio_pin_data.get_data()[-1]['BOARD'].keys())
}

print(board_to_tegra_soc)

# Create the I2C bus interface.
i2c_bus = busio.I2C("I2C1_SDA", "I2C1_SCL")

# Create a simple PCA9685 class instance.
pca = PCA9685(i2c_bus)
pca.frequency = 50

servo0 = servo.Servo(pca.channels[0])

servo0.angle = 90