from flask import Flask, render_template, Response
from json import loads
from flask_socketio import SocketIO, emit
import threading
from car import Car
from camera import Camera
from time import sleep
import regex as re

HOST = "0.0.0.0"
PORT = 8080
IN_MOTOR_DRIVER = {
    "ENA": "GPIO_PE6",
    "IN1": "DAP4_FS",
    "IN2": "UART2_CTS",
}
SERVO_PIN_PCA = 0

camera = Camera.get_instance()
print("Camera is opened: "+ str(camera.is_opened()))
if not camera.is_opened():
    raise Exception("Camera is not opened!")
app = Flask(__name__, template_folder='templates')
app.config["SECRET_KEY"] = 'secret!'
socketio = SocketIO(app)
car = Car(IN_MOTOR_DRIVER, SERVO_PIN_PCA)

@app.route('/')
def index():
    return render_template('index.html')
@app.route('/stream_camera')
def stream_camera():
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')
def gen():
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@socketio.on('message', namespace='/socket')
def message(data):
    print(data)
    # message = "forward 25"
    if re.match(r'forward \d+', data):
        speed = int(data.split(' ')[1])
        car.forward(speed)
    elif re.match(r'backward \d+', data):
        speed = int(data.split(' ')[1])
        car.backward(speed)
    elif data == 'left':
        car.turn_left(3)
    elif data == 'right':
        car.turn_right(3)
    elif data == 'stop':
        car.stop()
    
    

def main():
    capture_thread = threading.Thread(target=camera.update)
    capture_thread.daemon = True
    capture_thread.start()

    socketio.run(app, host=HOST, port=PORT)

if __name__ == '__main__':
    main()