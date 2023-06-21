import tensorrt as trt
import cv2
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
import time
from ultis import post_processing_yolo, generate_lines, show_img_with_boxes, show_img_with_lines, preprocess_ufld, preprocess_yolo
from concurrent.futures import ThreadPoolExecutor
import os

input_size_1 = 416
input_height_2 = 288
input_width_2 = 512
crop_raito = 0.42
griding_num = 100
cls_num_per_lane = 18
conf_thresh_1 = 0.5
nms_thresh_1 = 0.4

CLASSES = ['traffic light',
 'car',
 'person',
 'bus',
 'truck',
]

row_anchor = np.linspace(input_height_2*crop_raito, input_height_2-1, 18).astype(int)
diff = row_anchor[1:] - row_anchor[:-1]
diff = np.mean(diff)

# video_test_path = "img_test/example.mp4"
video_test_path = "video_example/05081544_0305"
video_save_path = "img_test/example_out_models.mp4"
model_1_path = "yolov4_engine.trt"
model_2_path = "culane_res18_engine.trt"

f_1 = open(model_1_path, "rb")
f_2 = open(model_2_path, "rb")
runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
engine_1 = runtime.deserialize_cuda_engine(f_1.read())
engine_2 = runtime.deserialize_cuda_engine(f_2.read())
context_1 = engine_1.create_execution_context()
context_2 = engine_2.create_execution_context()

output_loc_1 = np.empty((1, 2535, 1, 4), dtype=np.float16)
output_conf_1 = np.empty((1, 2535, len(CLASSES)), dtype=np.float16)
output_2 = np.empty((1, (griding_num+1), cls_num_per_lane, 4), dtype=np.float16)
# Allow engine to allocate memory and copy over data
d_input_1 = cuda.mem_alloc(1 * 3 * input_size_1 * input_size_1 * np.dtype(np.float16).itemsize)
d_input_2 = cuda.mem_alloc(1 * 3 * input_height_2 * input_width_2 * np.dtype(np.float16).itemsize)
d_output_loc_1 = cuda.mem_alloc(1 * output_loc_1.nbytes)
d_output_conf_1 = cuda.mem_alloc(1 * output_conf_1.nbytes)
d_output_2 = cuda.mem_alloc(1 * output_2.nbytes)

bindings_1 = [int(d_input_1), int(d_output_loc_1), int(d_output_conf_1)]
bindings_2 = [int(d_input_2), int(d_output_2)]

stream = cuda.Stream()

def predict_1(batch): # result gets copied into output
    # Transfer input data to device
    cuda.memcpy_htod_async(d_input_1, batch, stream)
    # Execute model
    context_1.execute_async_v2(bindings_1, stream.handle, None)
    # Transfer predictions back
    cuda.memcpy_dtoh_async(output_loc_1, d_output_loc_1, stream)
    cuda.memcpy_dtoh_async(output_conf_1, d_output_conf_1, stream)
    # Synchronize threads
    stream.synchronize()
    return output_loc_1, output_conf_1

def predict_2(batch): # result gets copied into output
    # Transfer input data to device
    cuda.memcpy_htod_async(d_input_2, batch, stream)
    # Execute model
    context_2.execute_async_v2(bindings_2, stream.handle, None)
    # Transfer predictions back
    cuda.memcpy_dtoh_async(output_2, d_output_2, stream)
    # Synchronize threads
    stream.synchronize()
    return output_2

# Load the video
width, height = 1640, 590
fps = 25
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out_video = cv2.VideoWriter(video_save_path, fourcc, fps, (width, height))

for files in os.listdir(video_test_path):
    file_path = os.path.join(video_test_path, files)
    frame = cv2.imread(file_path)
    start = time.time()
    thread_pool = ThreadPoolExecutor(max_workers=2)
    frame_1 = thread_pool.submit(preprocess_yolo, frame, input_size_1)
    frame_2 = thread_pool.submit(preprocess_ufld, frame, input_height_2, input_width_2)
    frame_1 = frame_1.result()
    frame_2 = frame_2.result()
    output_loc_1, output_conf_1 = predict_1(frame_1)
    lines_batch_2 = predict_2(frame_2)
    bbox_batch = thread_pool.submit(post_processing_yolo, conf_thresh_1, nms_thresh_1, output_loc_1, output_conf_1, height, width)
    lines_batch = thread_pool.submit(generate_lines, lines_batch_2, griding_num, input_height_2, input_width_2, height, width, diff)
    bbox_batch = bbox_batch.result()
    lines_batch = lines_batch.result()
    frame = show_img_with_boxes(frame, bbox_batch[0], CLASSES)
    frame = show_img_with_lines(frame, lines_batch[0])
    end = time.time()
    fps = 1/(end-start)
    print("FPS: {:.2f}".format(fps))
    cv2.putText(frame, "FPS: {:.2f}".format(fps), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    out_video.write(frame)