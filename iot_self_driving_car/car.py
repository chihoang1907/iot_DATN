from RPi import GPIO
from adafruit_servokit import ServoKit
import time

CENTER_SERVO = 95
MIN_SERVO = 65
MAX_SERVO = 135
TIME_SLEEP_SERVO = 0.05

MIN_DUTY_DC_RUN = 60
TIME_SLEEP_DC_RUN = 0.1

class Car:

    def __init__(self, in_motor_driver, servo_pin_pca, hz=100):
        # L298 
        self.in_motor_driver = in_motor_driver
        self.ena_pwm = None
        self.speed = 0
        self.hz = hz
        self.servo_pin_pca = servo_pin_pca
        self.deg = CENTER_SERVO
        self.servo= None
        self.setup_pin()
    
    def __del__(self):
        GPIO.cleanup()

    def setup_pin(self):
        GPIO.setmode(GPIO.TEGRA_SOC)
        # Setup motor driver
        GPIO.setup(self.in_motor_driver["IN1"], GPIO.OUT, initial=GPIO.LOW)
        GPIO.setup(self.in_motor_driver["IN2"], GPIO.OUT, initial=GPIO.LOW)
        GPIO.setup(self.in_motor_driver["ENA"], GPIO.OUT)
        self.ena_pwm = GPIO.PWM(self.in_motor_driver["ENA"], self.hz)
        self.ena_pwm.start(0)
        
        # Setup servo
        servoKit = ServoKit(channels=16)
        self.servo = servoKit.servo[self.servo_pin_pca]
        self.servo.angle = self.deg
        
    def forward(self, speed):
        GPIO.output(self.in_motor_driver["IN1"], GPIO.HIGH)
        GPIO.output(self.in_motor_driver["IN2"], GPIO.LOW)
        if self.speed == 0 or (self.speed < 0 and -self.speed < MIN_DUTY_DC_RUN):
            self.ena_pwm.ChangeDutyCycle(MIN_DUTY_DC_RUN)
            time.sleep(TIME_SLEEP_DC_RUN)
        # clip speed
        speed = max(min(speed, 100), 0)
        self.ena_pwm.ChangeDutyCycle(speed)
        self.speed = speed

    def stop(self):
        GPIO.output(self.in_motor_driver["IN1"], GPIO.LOW)
        GPIO.output(self.in_motor_driver["IN2"], GPIO.LOW)
        self.ena_pwm.ChangeDutyCycle(0)
        self.speed = 0
        
    def backward(self, speed):
        GPIO.output(self.in_motor_driver["IN1"], GPIO.LOW)
        GPIO.output(self.in_motor_driver["IN2"], GPIO.HIGH)
        if self.speed == 0 or (self.speed > 0 and self.speed < MIN_DUTY_DC_RUN):
            self.ena_pwm.ChangeDutyCycle(MIN_DUTY_DC_RUN)
            time.sleep(TIME_SLEEP_DC_RUN)
        # clip speed
        speed = max(min(speed, 100), 0)
        self.ena_pwm.ChangeDutyCycle(speed)
        self.speed = -speed

    def turn_left(self, deg_turn):
        self.deg = max(self.deg - deg_turn, MIN_SERVO)
        self.servo.angle = self.deg
        # time.sleep(TIME_SLEEP_SERVO)
    
    def turn_right(self, deg_turn):
        self.deg = min(self.deg + deg_turn, MAX_SERVO)
        self.servo.angle = self.deg
        # time.sleep(TIME_SLEEP_SERVO)

    def turn_corner(self, deg):
        # clip value
        deg = max(min(deg, MAX_SERVO), MIN_SERVO)
        self.deg = deg
        self.servo.angle = self.deg

    def is_stop(self):
        return self.speed == 0