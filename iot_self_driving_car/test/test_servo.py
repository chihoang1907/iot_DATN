# Python Script
# https://www.electronicshub.org/raspberry-pi-l298n-interface-tutorial-control-dc-motor-l298n-raspberry-pi/

import RPi.GPIO as GPIO

from time import sleep

servoPIN = 32
GPIO.setmode(GPIO.BOARD)
GPIO.setup(servoPIN, GPIO.OUT)

servo = GPIO.PWM(servoPIN, 50) # GPIO 32 for PWM with 50Hz

# min duty% = 0 degrees
# max duty% = 180 degrees
min_duty = 2.2
max_duty = 11.9

ratio = (max_duty - min_duty)*1.0/180

def deg_to_duty(deg):
    return deg*ratio + min_duty
servo.start(2.2) # Initialization
sleep(1)
servo.ChangeDutyCycle(11.9)
sleep(1)
min_deg_car = 50
max_deg_car = 180 - min_deg_car

# for i in range(90, 121, 10):
#     servo.ChangeDutyCycle(deg_to_duty(i))
#     print(i)
#     sleep(0.5)
# for i in range(180, -1, -10):
#     servo.ChangeDutyCycle(deg_to_duty(i))
#     print(i)
#     sleep(0.5)
# for i in range(0, 91, 10):
#     servo.ChangeDutyCycle(deg_to_duty(i))
#     print(i)
#     sleep(0.5)
servo.ChangeDutyCycle(deg_to_duty(90))
sleep(1)
GPIO.cleanup()
