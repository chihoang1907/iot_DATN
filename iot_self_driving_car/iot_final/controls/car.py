from RPi import GPIO
from adafruit_servokit import ServoKit
import time

class Car:

    def __init__(self, in_motor_driver, servo_pin_pca, cfg, hz=100):
        # L298 
        self.in_motor_driver = in_motor_driver
        self.ena_pwm = None
        self.speed = 0
        self.hz = hz
        self.servo_pin_pca = servo_pin_pca
        self.center_servo = cfg.CENTER_SERVO
        self.min_servo = cfg.MIN_SERVO
        self.max_servo = cfg.MAX_SERVO
        self.deg = self.center_servo
        self.min_duty_dc_run = cfg.MIN_DUTY_DC_RUN
        self.time_sleep_dc_run = cfg.TIME_SLEEP_DC_RUN
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
        if self.speed == 0 or (self.speed < 0 and -self.speed < self.min_duty_dc_run):
            self.ena_pwm.ChangeDutyCycle(self.min_duty_dc_run)
            time.sleep(self.time_sleep_dc_run)
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
        if self.speed == 0 or (self.speed > 0 and self.speed < self.min_duty_dc_run):
            self.ena_pwm.ChangeDutyCycle(self.min_duty_dc_run)
            time.sleep(self.time_sleep_dc_run)
        # clip speed
        speed = max(min(speed, 100), 0)
        self.ena_pwm.ChangeDutyCycle(speed)
        self.speed = -speed

    def turn_left(self, deg_turn):
        self.deg = max(self.deg - deg_turn, self.min_servo)
        self.servo.angle = self.deg
        # time.sleep(TIME_SLEEP_SERVO)
    
    def turn_right(self, deg_turn):
        self.deg = min(self.deg + deg_turn, self.max_servo)
        self.servo.angle = self.deg
        # time.sleep(TIME_SLEEP_SERVO)

    def turn_corner(self, deg):
        # clip value
        deg = max(min(deg, self.max_servo), self.min_servo)
        self.deg = deg
        self.servo.angle = self.deg

    def run(self):
        if self.speed != self.min_duty_dc_run:
            if self.speed != 0:
                self.forward(self.speed)
            else:
                self.forward(self.min_duty_dc_run)
                self.speed = self.min_duty_dc_run