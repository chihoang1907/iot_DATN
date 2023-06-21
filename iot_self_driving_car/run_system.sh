# setup pin PWM
# Pin 32
sudo busybox devmem 0x700031fc 32 0x45
sudo busybox devmem 0x6000d504 32 0x2
echo 0 > /sys/class/pwm/pwmchip0/export
echo 1 > /sys/class/pwm/pwmchip0/pwm0/enable
# Pin 33
sudo busybox devmem 0x70003248 32 0x46
sudo busybox devmem 0x6000d100 32 0x00
echo 2 > /sys/class/pwm/pwmchip0/export
echo 1 > /sys/class/pwm/pwmchip0/pwm2/enable

cd /home/jetson/Documents/iot_self_driving_car/
source .venv/bin/activate
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1
export OPENBLAS_CORETYPE=ARMV8
# python web_server.py