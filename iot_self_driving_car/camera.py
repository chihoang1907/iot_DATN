import tensorrt as trt
import cv2
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
import time
from LaneDetection import LaneDetection
from Yolo import Yolo
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

MODEL_LD_PATH = "ufld_mydataset_2_engine.trt"

#YOLO
CLASSES = [
    "object",
    "traffic light"
]
NUM_CLASS = len(CLASSES)
INPUT_HEIGHT_YOLO = 416
INPUT_WIDTH_YOLO = 416
MODEL_YOLO_PATH = "yolov4_mydataset_2_engine.trt"
NMS_THRESH = 0.4
CONF_THRESH = 0.85
CENTER_SERVO = 95


# Car
MIN_DUTY_DC_RUN = 48
TIME_SLEEP_DC_RUN = 0.1
ARRAY_DISTANCE = np.array([454, 376, 331, 292, 266]) # 10, 20, 30, 40 cm
ARRAY_WIDTH_LANE = np.array([546, 442, 378, 320, 288]) # 10, 20, 30, 40 cm
WIDTH_LANE=30 #cm

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

def convert_pixel_to_cm(distance, width_lane_px):
    return distance / width_lane_px * WIDTH_LANE

def get_distance_oriline_linedt(frame_recognized, arr_x_oriline, color_linedt):
    index_lanes = []
    for i in range(len(ARRAY_DISTANCE)):
        index_lane = np.where(np.all(frame_recognized[ARRAY_DISTANCE[i], :, :] == color_linedt, axis=-1))[0]
        index_lane = np.mean(index_lane) if len(index_lane) > 0 else -1
        index_lanes.append(index_lane)
    index_lanes = np.array(index_lanes)
    mask = index_lanes != -1
    distance = convert_pixel_to_cm(index_lanes[mask] - arr_x_oriline[mask], ARRAY_WIDTH_LANE[mask]) if len(mask) > 0 else []
    distance = np.mean(distance) if len(distance) > 0 else 0
    return distance

class Camera(object):
    _instance = None
    @staticmethod
    def get_instance():
        if Camera._instance is None:
            Camera._instance = Camera()
        return Camera._instance
    def __init__(self, car):
        if Camera._instance != None:
            raise Exception("This class is a singleton!")
        else:
            Camera._instance = self
            self.video = cv2.VideoCapture(0, cv2.CAP_V4L)
            self.ori_width = 640
            self.ori_height = 480
            self.center_width = self.ori_width // 2
            self.video.set(cv2.CAP_PROP_FRAME_WIDTH, self.ori_width)
            self.video.set(cv2.CAP_PROP_FRAME_HEIGHT, self.ori_height)
            self.video.set(cv2.CAP_PROP_FPS, 30)
            self.grabbed, self.frame = self.video.read()

            self.stream = cuda.Stream()
            self.car = car

            self.frame_recognized = None
            self.lines = None
            self.objects = None
            self.thread_pool = ThreadPoolExecutor(max_workers=2)

            self.model_LD = LaneDetection(MODEL_LD_PATH, self.ori_height, self.ori_width, INPUT_HEIGHT_LD, INPUT_WIDTH_LD, CROP_RAITO, CLS_NUM_PER_LANE, 
                                          GRIDING_NUM, NUM_LANE, self.stream)
            self.model_YOLO = Yolo(MODEL_YOLO_PATH, self.ori_height, self.ori_width, INPUT_HEIGHT_YOLO, INPUT_WIDTH_YOLO, CLASSES,
                                   NMS_THRESH, CONF_THRESH, self.stream)
            
            self.arr_x_oriline_right = []
            self.arr_x_oriline_left = []
            for i in range(len(ARRAY_DISTANCE)):
                width_lane = ARRAY_WIDTH_LANE[i]
                distance = ARRAY_DISTANCE[i]
                x1 = self.center_width - width_lane // 2
                x2 = self.center_width + width_lane // 2
                self.arr_x_oriline_right.append(x2)
                self.arr_x_oriline_left.append(x1)
            self.arr_x_oriline_right = np.array(self.arr_x_oriline_right)
            self.arr_x_oriline_left = np.array(self.arr_x_oriline_left)

            self.time_turn_limmit = 2
            self.start_turn = None
            self.object_before = None
    def draw_line_limmit(self, frame_recognized):
        dis_step = 10 #cm
        # draw line
        for i in range(len(ARRAY_DISTANCE)):
            distance = ARRAY_DISTANCE[i]
            y = distance
            cv2.line(frame_recognized, (0, y), (self.ori_width, y), (0, 255, 0), 1)
            # cv2.putText(frame_recognized, "{}cm".format(dis_step*(i+1)), (x1, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        return frame_recognized
            
    def __del__(self):
        self.video.release()

    def get_frame(self):
        _, jpeg=cv2.imencode('.jpg',self.frame)
        return jpeg.tobytes()
    
    def get_frame_recognized(self):
        _, jpeg=cv2.imencode('.jpg',self.frame_recognized)
        return jpeg.tobytes()
    
    def update(self):
        while True:
            (self.grabbed, self.frame) = self.video.read()
            if self.grabbed:
                self.recorgnize()
            else:
                raise Exception("Cannot read frame")

    def capture(self):
        return self.frame
    
    def is_opened(self):
        return self.video.isOpened()
    
    def recorgnize(self):
        start = time.time()
        frame_recognized = self.frame.copy()
        img_preprocess_LD = self.thread_pool.submit(self.model_LD.preprocess, frame_recognized)
        img_preprocess_YOLO = self.thread_pool.submit(self.model_YOLO.preprocess, frame_recognized)
        img_preprocess_LD = img_preprocess_LD.result()
        img_preprocess_YOLO = img_preprocess_YOLO.result()
        
        self.lines = self.model_LD.get_lane(img_preprocess_LD)
        output_loc, output_conf = self.model_YOLO.predict(img_preprocess_YOLO)

        thread_draw = self.thread_pool.submit(self.draw_line_limmit, frame_recognized)
        thread_yolo = self.thread_pool.submit(self.model_YOLO.get_objects, output_loc, output_conf)
        self.objects = thread_yolo.result()
        self.frame_recognized = thread_draw.result()

        self.frame_recognized = draw_line_lane(self.frame_recognized, self.arr_x_oriline_left, self.arr_x_oriline_right)
        self.frame_recognized = self.model_YOLO.get_objects_img(self.frame_recognized, self.objects)
        self.frame_recognized = self.model_LD.get_lane_img(self.frame_recognized, self.lines)

        line_left = self.lines[0]
        line_right = self.lines[2]
        line_left_1 = self.lines[1]
        # get line from 10cm to 40cm
        index_dis = 3
        # line_left = line_left[line_left[:, 1] <= ARRAY_DISTANCE[0]]
        # line_left = line_left[line_left[:, 1] >= ARRAY_DISTANCE[index_dis]]
        # line_right = line_right[line_right[:, 1] <= ARRAY_DISTANCE[0]]
        # line_right = line_right[line_right[:, 1] >= ARRAY_DISTANCE[index_dis]]

        if len(self.objects) > 0:
            # get object in range from 0cm to 30cm
            objects = self.objects[self.objects[:, 3] >= ARRAY_DISTANCE[3]]
            if len(objects) > 0:
                # get object min distance
                object_min = objects[np.argmax(objects[:, 3])]
                index_dis_obj= np.where(np.logical_and(ARRAY_DISTANCE >= object_min[1], ARRAY_DISTANCE <= object_min[3]))[0][0]
                # check object in lane
                if object_min[0] >= self.arr_x_oriline_left[index_dis_obj] and object_min[2] <= self.arr_x_oriline_right[index_dis_obj]:
                    self.start_turn = time.time()
                else:
                    dis_obj_left = np.abs(object_min[0] - self.arr_x_oriline_left[index_dis_obj])
                    dis_obj_right = np.abs(self.arr_x_oriline_right[index_dis_obj] - object_min[2])
                    # get max distance
                    if dis_obj_left > dis_obj_right and dis_obj_left >= 0.7 * ARRAY_WIDTH_LANE[index_dis_obj]:
                        # turn left
                        dis = np.abs(object_min[0] - self.arr_x_oriline_right[index_dis_obj])
                        dis = convert_pixel_to_cm(dis, index_dis_obj)
                        # self.car.turn_corner(CENTER_SERVO - dis)
                        # if self.car.is_stop():
                        #     self.car.forward(MIN_DUTY_DC_RUN)
                        return self.frame_recognized
                    elif dis_obj_right  >= 0.7 * ARRAY_WIDTH_LANE[index_dis_obj]:
                        # turn left
                        dis = np.abs(object_min[2] - self.arr_x_oriline_left[index_dis_obj])
                        dis = convert_pixel_to_cm(dis, index_dis_obj)
                        # self.car.turn_corner(CENTER_SERVO + dis)
                        # if self.car.is_stop():
                        #     self.car.forward(MIN_DUTY_DC_RUN)
                        return self.frame_recognized
                    else:
                        # stop
                        self.car.stop()
                        print("stop")
            
        if self.start_turn and time.time() - self.start_turn < self.time_turn_limmit:
            # change lane
            if len(line_left_1) > 2:
                name_linedt = "left_1"
                linedt = line_left_1
                colordt = (255, 255, 0)
                arr_x_oriline = self.arr_x_oriline_left
            elif len(line_left) > 2:
                name_linedt = "left"
                linedt = line_left
                colordt = (255, 0, 0)
                arr_x_oriline = self.arr_x_oriline_right
            else:
                # stop
                self.car.stop()
                return self.frame_recognized
            print("change lane")
        else:
            self.start_turn = None
            if len(line_right) > 2:
                name_linedt = "right"
                linedt = line_right
                colordt = (0, 0, 255)
                arr_x_oriline = self.arr_x_oriline_right
            elif len(line_left) > 2:
                name_linedt = "left"
                linedt = line_left
                colordt = (255, 0, 0)
                arr_x_oriline = self.arr_x_oriline_left
            else:
                # stop
                self.car.stop()
                return self.frame_recognized
        distance = get_distance_oriline_linedt(frame_recognized, arr_x_oriline, colordt)
        thresh_dis = 2
        # smooth
        linedt_smooth = smooth_line(linedt)
        vector_oriline = np.array([arr_x_oriline[index_dis] - arr_x_oriline[0], ARRAY_DISTANCE[index_dis] - ARRAY_DISTANCE[0]])
        # cross
        vector_linedt = linedt_smooth[2:] - linedt_smooth[:2]
        # if np.abs(linedt_smooth[3] - arr_x_oriline[index_dis]) < np.abs(linedt_smooth[3] - self.center_width):
        #     vector_oriline = np.array([0, -1])
        #     cross = np.cross(vector_oriline, vector_linedt)
        #     vector_oriline = np.array([arr_x_oriline[index_dis] - arr_x_oriline[0], ARRAY_DISTANCE[index_dis] - ARRAY_DISTANCE[0]]) * (-1 if cross > 0 else 1)
        angle = get_angle(vector_oriline, vector_linedt)
        if np.abs(distance) >= thresh_dis:
            angle = np.abs(angle) * (1 if distance > 0 else -1)
        
        angle_dis = (distance) * 0.8 if np.abs(distance) > thresh_dis else 0
        angle = angle * 0.8 if np.abs(angle) > 2 else 0 
        angle = angle #if distance > 0 else -angle
        angle_turn = int(angle + angle_dis)
        print(distance, angle, angle_dis, angle_turn)
        # self.car.turn_corner(CENTER_SERVO + angle_turn)
        # if self.car.is_stop():
        #     self.car.forward(MIN_DUTY_DC_RUN)

        end = time.time()
        fps = 1 / (end - start)
        cv2.putText(self.frame_recognized, "FPS: {:.2f}".format(fps), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        return self.frame_recognized
    
    def control(self):
        index_lane_left = 0
        index_lane_right = 2

        distance_10cm = self.ori_height - ARRAY_DISTANCE[0]
        distance_20cm = self.ori_height - ARRAY_DISTANCE[1]
        distance_30cm = self.ori_height - ARRAY_DISTANCE[2]
        point_0 = np.array([0, self.ori_width // 2])