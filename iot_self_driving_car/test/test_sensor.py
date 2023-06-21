import RPi.GPIO as GPIO
from time import sleep

sensorPin = 26
GPIO.setmode(GPIO.BOARD)

GPIO.setup(sensorPin, GPIO.IN)

while True:
    print(GPIO.input(sensorPin))
    sleep(0.5)

GPIO.cleanup()