import tensorrt as trt
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
cls_num_per_lane = 30
num_lane = 3

row_anchor = np.linspace(input_height*crop_raito, input_height-1, cls_num_per_lane).astype(int)
diff = row_anchor[1:] - row_anchor[:-1]
diff = np.mean(diff)
col_sample = np.linspace(0, input_width - 1, griding_num)
col_sample_w = col_sample[1] - col_sample[0]

model_path = "ufld_mydataset_engine.trt"

f = open(model_path, "rb")
runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
engine = runtime.deserialize_cuda_engine(f.read())
context = engine.create_execution_context()

output = np.empty((1, num_lane, cls_num_per_lane, 2), dtype=np.float16)
# Allow engine to allocate memory and copy over data
d_input = cuda.mem_alloc(1 * 3 * input_height * input_width * np.dtype(np.float16).itemsize)
d_output = cuda.mem_alloc(output.nbytes)

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

img_path = "img_test/2ba7a824-capture_-_2023-06-24T202019.087.jpg"
img = cv2.imread(img_path)
img_copy = img.copy()
ori_height, ori_width, _ = img.shape
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# img = img_transforms(img)
img = cv2.resize(img, (input_width, input_height))
img = np.array(img)
img = img / 255.0
img = np.transpose(img, (2, 0, 1))
img = np.expand_dims(img, axis=0).astype(np.float16)
img = np.ascontiguousarray(img)
out = predict(img)

# lines = generate_lines(out[0], griding_num, input_height, input_width, ori_height, ori_width, diff)
# num_lane x num_rows x 2 (x, y) x index_rows, y location
lines = out[0]

lines_list = []
for line in lines:
    # filter out location =0
    line = line[line[:, 0] != 0]
    if line.shape[0] == 0:
        lines_list.append(line)
        continue
    line[:,0] = ((line[:, 0] * col_sample_w * ori_width / input_width) - 1).astype(int)
    line[:,1] = (ori_height - line[:, 1] * diff * ori_height / input_height).astype(int)
    lines_list.append(line)

colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255,255,0)]
for i in range(len(lines_list)):
    line = lines_list[i]
    for p in line:
        cv2.circle(img_copy, (int(p[0]), int(p[1])), 5, colors[i], -1)

cv2.imwrite("test.jpg", img_copy)