import numpy as np
import cv2
import time
import math
from Yolo import Yolo
from LaneDetection import LaneDetection
import pycuda.driver as cuda
from concurrent.futures import ThreadPoolExecutor

# LD
INPUT_HEIGHT_LD = 288
INPUT_WIDTH_LD = 512
CROP_RAITO = 0.42
GRIDING_NUM = 100
CLS_NUM_PER_LANE = 30
NUM_LANE = 3
ROW_ANCHOR = np.linspace(INPUT_HEIGHT_LD*CROP_RAITO, INPUT_HEIGHT_LD-1, CLS_NUM_PER_LANE).astype(int)
DIFF = ROW_ANCHOR[1:] - ROW_ANCHOR[:-1]
DIFF = np.mean(DIFF)
COL_SAMPLE = np.linspace(0, INPUT_WIDTH_LD - 1, GRIDING_NUM)
COL_SAMPLE_W = COL_SAMPLE[1] - COL_SAMPLE[0]

MODEL_LD_PATH = "ufld_mydataset_engine.trt"

#YOLO
CLASSES = [
    "object",
    "traffic light"
]
NUM_CLASS = len(CLASSES)
INPUT_HEIGHT_YOLO = 416
INPUT_WIDTH_YOLO = 416
MODEL_YOLO_PATH = "yolov4_my_dataset_engine.trt"
NMS_THRESH = 0.4
CONF_THRESH = 0.5

# camera
WIDTH_LANE=30 #cm
ARRAY_DISTANCE = np.array([455, 375, 330, 290, 268]) # 10, 20, 30, 40 cm
ARRAY_WIDTH_LANE = np.array([544, 440, 370, 310, 272]) # 10, 20, 30, 40 cm

ori_height = 480
ori_width = 640
center_width = ori_width // 2

stream = cuda.Stream()

model_LD = LaneDetection(MODEL_LD_PATH, ori_height, ori_width, INPUT_HEIGHT_LD, INPUT_WIDTH_LD, CROP_RAITO, CLS_NUM_PER_LANE, 
                                          GRIDING_NUM, NUM_LANE, stream)
model_YOLO = Yolo(MODEL_YOLO_PATH, ori_height, ori_width, INPUT_HEIGHT_YOLO, INPUT_WIDTH_YOLO, CLASSES,
                                   NMS_THRESH, CONF_THRESH, stream)
thread_pool = ThreadPoolExecutor(max_workers=2)

def draw_line_limmit(frame_recognized):
    dis_step = 10 #cm
    arr_x1 = []
    arr_x2 = []
    # draw line
    for i in range(len(ARRAY_DISTANCE)):
        width_lane = ARRAY_WIDTH_LANE[i]
        distance = ARRAY_DISTANCE[i]
        x1 = center_width - width_lane // 2
        x2 = center_width + width_lane // 2
        y = distance
        cv2.line(frame_recognized, (0, y), (ori_width, y), (0, 255, 0), 1)
        # cv2.putText(frame_recognized, "{}cm".format(dis_step*(i+1)), (x1, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        arr_x1.append(x1)
        arr_x2.append(x2)
    arr_x1 = np.array(arr_x1)
    arr_x2 = np.array(arr_x2)
    return frame_recognized, arr_x1, arr_x2

def draw_line_lane(frame_recognized, arr_x1, arr_x2):
    # draw green line
    n_point = 3
    # draw red line
    for i in range(len(ARRAY_DISTANCE) - 1):
        linspace_width_1 = np.linspace(0, ARRAY_WIDTH_LANE[i], n_point)
        linspace_width_2 = np.linspace(0, ARRAY_WIDTH_LANE[i+1], n_point)
        for j in range(n_point):
            x1 = arr_x1[i] + int(linspace_width_1[j])
            x2 = arr_x1[i+1] + int(linspace_width_2[j])
            y1 = ARRAY_DISTANCE[i]
            y2 = ARRAY_DISTANCE[i+1]
            cv2.line(frame_recognized, (x1, y1), (x2, y2), (0, 255, 0), 1)
    arr_x1 = np.array(arr_x1)
    arr_x2 = np.array(arr_x2)
    return frame_recognized

def smooth_line(line):
    if len(line) <= 2:
        return line
    index_point_center_right = len(line) // 2
    left_point = line[:index_point_center_right]
    right_point = line[index_point_center_right:]
    left_point = np.mean(left_point, axis=0)
    right_point = np.mean(right_point, axis=0)
    line = np.concatenate((left_point, right_point), axis=0)
    return line

def get_angle(vector1, vector2):
    cross = np.cross(vector1, vector2)
    angle = np.arcsin(cross/(np.linalg.norm(vector1)*np.linalg.norm(vector2)))
    angle = np.degrees(angle)
    return angle

def get_distance_oriline_linedt(frame_recognized, arr_x_oriline, color_linedt):
    index_lanes = []
    for i in range(len(ARRAY_DISTANCE)):
        index_lane = np.where(np.all(frame_recognized[ARRAY_DISTANCE[i], :, :] == color_linedt, axis=-1))[0]
        index_lane = np.mean(index_lane) if len(index_lane) > 0 else -1
        index_lanes.append(index_lane)
    index_lanes = np.array(index_lanes)
    mask = index_lanes != -1
    # distances_left = (index_lanes[mask_left, 0] - arr_x1[mask_left])/ARRAY_WIDTH_LANE[mask_left] * WIDTH_LANE if len(mask_left) > 0 else []
    distance = (index_lanes[mask] - arr_x_oriline[mask])/ARRAY_WIDTH_LANE[mask] * WIDTH_LANE if len(mask) > 0 else []
    distance = np.mean(distance) if len(distance) > 0 else 0
    return distance


img_path = "img_test/capture - 2023-06-24T202208.328.jpg"
frame = cv2.imread(img_path)

frame_recognized = frame.copy()
img_preprocess_LD = thread_pool.submit(model_LD.preprocess, frame_recognized)
img_preprocess_YOLO = thread_pool.submit(model_YOLO.preprocess, frame_recognized)
img_preprocess_LD = img_preprocess_LD.result()
img_preprocess_YOLO = img_preprocess_YOLO.result()

lines = model_LD.get_lane(img_preprocess_LD)
output_loc, output_conf = model_YOLO.predict(img_preprocess_YOLO)
thread_draw = thread_pool.submit(draw_line_limmit, frame_recognized)
thread_yolo = thread_pool.submit(model_YOLO.get_objects, output_loc, output_conf)
objects = thread_yolo.result()
frame_recognized, arr_x1, arr_x2 = thread_draw.result()

start = time.time()
frame_recognized = draw_line_lane(frame_recognized, arr_x1, arr_x2)
frame_recognized = model_LD.get_lane_img(frame_recognized, lines)

line_left = lines[0]
line_right = lines[2]
if len(line_left) > 2:
    linedt = line_left
    colordt = (255, 0, 0)
    arr_x_oriline = arr_x1
elif len(line_right) > 2:
    linedt = line_right
    colordt = (0, 0, 255)
    arr_x_oriline = arr_x2
else:
    # stop
    pass

distance = get_distance_oriline_linedt(frame_recognized, arr_x_oriline, colordt)
print("distance: ", distance)

# smooth
linedt_smooth = smooth_line(linedt)
# cross
vector_linedt = linedt_smooth[2:] - linedt_smooth[:2]
vector_oriline = np.array([arr_x_oriline[-1] - arr_x_oriline[0], ARRAY_DISTANCE[-1] - ARRAY_DISTANCE[0]])
angle = get_angle(vector_oriline, vector_linedt)
print("angle: ", angle)

end = time.time()
print("time: ", end - start)
# get distance from index[:,0] to x1
# distance = []
# for i in range(len(ARRAY_DISTANCE)):
#     if len(index_lanes_left[i]) > 0:
#         dis = (index_lanes_left[i][0] - arr_x2[i])/ARRAY_WIDTH_LANE[i] * WIDTH_LANE
#         distance.append(dis)
# mean_dis = np.mean(distance)
# print("distance: ", mean_dis)
# thresold = 2
# if np.abs(mean_dis) > thresold:
#     print("warning")
#     if mean_dis > 0:
#         # turn left
#         pass
#     else:
#         # turn right
#         pass



# frame_recognized = model_YOLO.get_objects_img(frame_recognized, objects)

cv2.imwrite("recognized.jpg", frame_recognized)