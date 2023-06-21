import tensorrt as trt
import cv2
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
import time

video_test_path = "example.mp4"
video_save_path = "example_out.mp4"
model_path = "yolov4_engine.trt"
NUM_CLASS = 5
CLASSES = ['traffic light',
 'car',
 'person',
 'bus',
 'truck',
]

img = cv2.imread("0000f77c-6257be58.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (416, 416))
img = img.astype(np.float16)
img = img / 255.0
img = np.transpose(img, (2, 0, 1))
img = np.expand_dims(img, axis=0)
img = np.ascontiguousarray(img)

f = open(model_path, "rb")
runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
engine = runtime.deserialize_cuda_engine(f.read())
context = engine.create_execution_context()

output_loc = np.empty((1, 2535, 1, 4), dtype=np.float16)
output_conf = np.empty((1, 2535, NUM_CLASS), dtype=np.float16)

# Allocate device memory
d_input = cuda.mem_alloc(1 * img.nbytes)
d_output_loc = cuda.mem_alloc(1 * output_loc.nbytes)
d_output_conf = cuda.mem_alloc(1 * output_conf.nbytes)

bindings = [int(d_input), int(d_output_loc), int(d_output_conf)]

stream = cuda.Stream()

def predict(batch): # result gets copied into output
    # Transfer input data to device
    cuda.memcpy_htod_async(d_input, batch, stream)
    # Execute model
    context.execute_async_v2(bindings, stream.handle, None)
    # Transfer predictions back
    cuda.memcpy_dtoh_async(output_loc, d_output_loc, stream)
    cuda.memcpy_dtoh_async(output_conf, d_output_conf, stream)
    # Syncronize threads
    stream.synchronize()

    return output_loc, output_conf

# Load the video
cap = cv2.VideoCapture(video_test_path)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out_video = cv2.VideoWriter(video_save_path, fourcc, fps, (width, height))

def nms_cpu(boxes, confs, nms_thresh=0.5, min_mode=False):
    # print(boxes.shape)
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    order = confs.argsort()[::-1]

    keep = []
    while order.size > 0:
        idx_self = order[0]
        idx_other = order[1:]

        keep.append(idx_self)

        xx1 = np.maximum(x1[idx_self], x1[idx_other])
        yy1 = np.maximum(y1[idx_self], y1[idx_other])
        xx2 = np.minimum(x2[idx_self], x2[idx_other])
        yy2 = np.minimum(y2[idx_self], y2[idx_other])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h

        if min_mode:
            over = inter / np.minimum(areas[order[0]], areas[order[1:]])
        else:
            over = inter / (areas[order[0]] + areas[order[1:]] - inter)

        inds = np.where(over <= nms_thresh)[0]
        order = order[inds + 1]
    
    return np.array(keep)

def post_processing(conf_thresh, nms_thresh, box_array, confs):
    num_classes = confs.shape[2]

    # [batch, num, 4]
    box_array = box_array[:, :, 0]

    # [batch, num, num_classes] --> [batch, num]
    max_conf = np.max(confs, axis=2)
    max_id = np.argmax(confs, axis=2)

    bboxes_batch = []
    for i in range(box_array.shape[0]):

        argwhere = max_conf[i] > conf_thresh
        l_box_array = box_array[i, argwhere, :]
        l_max_conf = max_conf[i, argwhere]
        l_max_id = max_id[i, argwhere]

        bboxes = []
        # nms for each class
        for j in range(num_classes):

            cls_argwhere = l_max_id == j
            ll_box_array = l_box_array[cls_argwhere, :]
            ll_max_conf = l_max_conf[cls_argwhere]
            ll_max_id = l_max_id[cls_argwhere]
            keep = nms_cpu(ll_box_array, ll_max_conf, conf_thresh, nms_thresh)

            if (keep.size > 0):
                ll_box_array = ll_box_array[keep, :]
                ll_max_conf = ll_max_conf[keep]
                ll_max_id = ll_max_id[keep]

                for k in range(ll_box_array.shape[0]):
                    bboxes.append([ll_box_array[k, 0], ll_box_array[k, 1], ll_box_array[k, 2], ll_box_array[k, 3], ll_max_conf[k], ll_max_id[k]])

        bboxes_batch.append(bboxes)
    return bboxes_batch

# Do inference
while cap.isOpened():
    start = time.time()
    ret, frame = cap.read()
    if not ret:
        break
    img_resize = cv2.resize(frame, (416, 416), interpolation=cv2.INTER_LINEAR)
    img = cv2.cvtColor(img_resize, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float16)
    img = img / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    img = np.ascontiguousarray(img)

    output_loc, output_conf = predict(img)

    height, width, _ = frame.shape
    bbox_batch = post_processing(0.4, 0.6, output_loc, output_conf)

    for box in bbox_batch[0]:
        x1 = int(box[0] * width)
        y1 = int(box[1] * height)
        x2 = int(box[2] * width)
        y2 = int(box[3] * height)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0))
        cv2.putText(frame, CLASSES[box[5]], (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    end = time.time()
    cv2.putText(frame, "FPS: {:.2f}".format(1 / (end - start)), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    # Save the frame
    out_video.write(frame)