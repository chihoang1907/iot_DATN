import tensorrt as trt
from scipy.special import softmax
import cv2
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
import time

input_height = 288
input_width = 512
crop_raito = 0.42
# ori_height = 590
# ori_width = 1640
griding_num = 100
cls_num_per_lane = 18

row_anchor = np.linspace(input_height*crop_raito, input_height-1, 18).astype(int)
diff = row_anchor[1:] - row_anchor[:-1]
diff = np.mean(diff)

video_test_path = "example.mp4"
video_save_path = "example_out_ld_ufld.mp4"
model_path = "culane_res18_engine.trt"

f = open(model_path, "rb")
runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
engine = runtime.deserialize_cuda_engine(f.read())
context = engine.create_execution_context()

output = np.empty((1, (griding_num+1), cls_num_per_lane, 4), dtype=np.float16)
# Allow engine to allocate memory and copy over data
d_input = cuda.mem_alloc(1 * 3 * input_height * input_width * np.dtype(np.float16).itemsize)
d_output = cuda.mem_alloc(1 * cls_num_per_lane * (griding_num+1) * 4 * np.dtype(np.float16).itemsize)

bindings = [int(d_input), int(d_output)]

stream = cuda.Stream()

def predict(batch): # result gets copied into output
    # Transfer input data to device
    cuda.memcpy_htod_async(d_input, batch, stream)
    # Execute model
    context.execute_async_v2(bindings, stream.handle, None)
    # Transfer predictions back
    cuda.memcpy_dtoh_async(output, d_output, stream)
    # Syncronize threads
    stream.synchronize()
    return output

def generate_lines(out, griding_num, input_height, input_width, ori_height, ori_width, diff):

    col_sample = np.linspace(0, input_width - 1, griding_num)
    col_sample_w = col_sample[1] - col_sample[0]

    lines = []
    for j in range(out.shape[0]):
        out_j = out[j]
        out_j = out_j[:, ::-1, :]
        prob = softmax(out_j[:-1, :, :], axis=0)
        idx = np.arange(griding_num) + 1
        idx = idx.reshape(-1, 1, 1)
        loc = np.sum(prob * idx, axis=0)
        out_j = np.argmax(out_j, axis=0)
        loc[out_j == griding_num] = 0
        out_j = loc

        for i in range(out_j.shape[1]):
            line = []
            if np.sum(out_j[:, i] != 0) > 2:
                for k in range(out_j.shape[0]):
                    if out_j[k, i] > 0:
                        line.append(int(out_j[k, i] * col_sample_w * ori_width / input_width) - 1)
                        line.append(int(ori_height - k * diff * ori_height / input_height) - 1)
            lines.append(line)
        return lines

def show_img_with_lines(img, lines):
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255,255,0)]
    for i in range(len(lines)):
        line = lines[i]
        for p in range(0, len(line), 2):
            cv2.circle(img, (line[p], line[p+1]), 5, colors[i], -1)
    return img
def foward(img):
    ori_height, ori_width, _ = img.shape
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # img = img_transforms(img)
    img = cv2.resize(img, (input_width, input_height))
    img = np.array(img)
    img = img / 255.0
    img = np.transpose(img, (2, 0, 1))
    # Normalize (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    img[0] = (img[0] - 0.485)/0.229
    img[1] = (img[1] - 0.456)/0.224
    img[2] = (img[2] - 0.406)/0.225
    img = np.expand_dims(img, axis=0).astype(np.float16)
    img = np.ascontiguousarray(img)
    out = predict(img)

    lines = generate_lines(out, griding_num, input_height, input_width, ori_height, ori_width, diff)
    return lines

# Load the video
cap = cv2.VideoCapture(video_test_path)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out_video = cv2.VideoWriter(video_save_path, fourcc, fps, (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        start = time.time()
        lines = foward(frame)
        frame = show_img_with_lines(frame, lines)
        end = time.time()
        cv2.putText(frame, "FPS: {:.2f}".format(1/(end-start)), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        out_video.write(frame)
    else:
        break