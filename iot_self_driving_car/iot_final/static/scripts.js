let speed = document.getElementById('speed')
let speed_value = document.getElementById('speed_value')
let corner = document.getElementById('corner')
let corner_value = document.getElementById('corner_value')
let status_control = document.getElementById("status_control")
let status_control_value = document.getElementById("status_control_value")
let status_car = document.getElementById("status_car")

speed_value.innerHTML = speed.value
corner_value.innerHTML = corner.value
status_control_value.innerHTML = status_control.checked ? 'Turn On' : 'Turn Off'
status_car.innerHTML = "Stop"

// Connect socket
var socket = io.connect('http://' + location.host + '/socket')
socket.on('connect', function() {
    console.log('Connected');
});
// Listen for messages
socket.on('message', function(data) {
    console.log('Incoming message:', data);
    data = data.split(" ")
    let action = data[0];
    let value = parseInt(data[1]);
    // corner
    if(action == "corner"){
        corner.value = value;
        corner_value.innerHTML = corner.value;
    }
    if(action == "forward" || action == "backward"){
        status_car.innerHTML = action;
        speed.value = value;
        speed_value.innerHTML = speed.value;
    }
    if(action == "stop"){
        status_car.innerHTML = action;
    }
    if(action == "status_control"){
        status_control.checked = Boolean(value);
        status_control_value.innerHTML = status_control.checked ? 'Turn On' : 'Turn Off'
        if (status_control.checked){
            status_car.innerHTML = "Auto"
        }
    }
});
// keypress event
document.onkeypress = function(event) {
    var key_press = String.fromCharCode(event.keyCode).toLocaleLowerCase();
    console.log(key_press);
    if(key_press == 'w'){
        socket.emit('message', 'speed '+parseInt(speed.value));
    }
    if(key_press == 's'){
        socket.emit('message', 'speed '+parseInt(-speed.value));
    }
    if(key_press == 'a'){
        socket.emit('message', 'corner '+(parseInt(corner.value)-3));
    }
    if(key_press == 'd'){
        socket.emit('message', 'corner '+(parseInt(corner.value)+3));
    }
    if(key_press == 'q'){
        socket.emit('message', 'stop 0');
    }
};
capture = function(){
    // save frame in tag img
    var frame = document.getElementById("frame");
    var canvas = document.createElement("canvas");
    canvas.width = frame.width;
    canvas.height = frame.height;
    var ctx = canvas.getContext("2d");
    ctx.drawImage(frame, 0, 0, frame.width, frame.height);
    var dataURI = canvas.toDataURL("image/jpeg");
    var save = document.createElement("a");
    save.href = dataURI;
    save.download = "capture.jpg";
    save.click();
}
onchangeSpeed = function(){
    speed_value.innerHTML = speed.value;
}
onchangeCorner = function(){
    console.log("change")
    corner_value.innerHTML = corner.value;
    socket.emit('message', 'corner '+corner.value);
}
onchangeStatusControl = function(){
    status_control_value.innerHTML = status_control.checked ? 'Turn On' : 'Turn Off';
    socket.emit('message', 'status_control '+(+status_control.checked));
    if (!status_control.checked){
        socket.emit('message', 'stop 0');
    }
}