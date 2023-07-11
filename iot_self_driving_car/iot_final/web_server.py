from flask import Flask, render_template, Response
from json import loads
from flask_socketio import SocketIO, emit
import threading
from controls.camera import Camera
from controls.car import Car
from time import sleep
import regex as re
import configs.configs as cfg

HOST = "0.0.0.0"
PORT = 8080
IN_MOTOR_DRIVER = {
    "ENA": "GPIO_PE6",
    "IN1": "DAP4_FS",
    "IN2": "UART2_CTS",
}
SERVO_PIN_PCA = 0

app = Flask(__name__, template_folder='templates')
app.config["SECRET_KEY"] = 'secret!'
socketio = SocketIO(app)
car = Car(IN_MOTOR_DRIVER, SERVO_PIN_PCA, cfg)
camera = Camera(car, cfg)
camera.status_control = True
print("Camera is opened: "+ str(camera.is_opened()))
if not camera.is_opened():
    raise Exception("Camera is not opened!")

@app.route('/')
def index():
    return render_template('index.html')
@app.route('/stream_camera')
def stream_camera():
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')
def gen():
    while True:
        frame = camera.get_frame_recognized()
        yield (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@socketio.on('message', namespace='/socket')
def message(data):
    print(data)
    try:
        data = data.split(" ")
        action = data[0]
        value = int(data[1])
    except:
        return

    if action == 'speed':
        if value >= 0:
            action = "forward"
            car.forward(value)
        else:
            action = "backward"
            value = -value
            car.backward(value)
    elif action == 'corner':
        car.turn_corner(value)
        value = car.deg
    elif action == 'stop':
        car.stop()
    elif action == 'status_control':
        camera.status_control = value
    emit('message', action + " " + str(value), namespace='/socket',broadcast=True)

def main():
    thread_web = threading.Thread(target=socketio.run, args=(app, HOST, PORT))
    thread_web.daemon = True
    thread_web.start()
    while camera.is_opened():
        camera.update()

if __name__ == '__main__':
    main()