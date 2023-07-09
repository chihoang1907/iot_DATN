import tensorrt as trt
import cv2
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
import time
from models.LaneDetection import LaneDetection
from models.Yolo import Yolo
from concurrent.futures import ThreadPoolExecutor

def draw_line_lane(frame_recognized, ARRAY_DISTANCE, ARRAY_WIDTH_LANE, arr_x1, arr_x2):
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

def convert_pixel_to_cm(distance, width_lane_px, WIDTH_LANE):
    return distance / width_lane_px * WIDTH_LANE

def get_distance_oriline_linedt(frame_recognized, ARRAY_DISTANCE, ARRAY_WIDTH_LANE, WIDTH_LANE, arr_x_oriline, color_linedt):
    index_lanes = []
    for i in range(len(ARRAY_DISTANCE)):
        index_lane = np.where(np.all(frame_recognized[ARRAY_DISTANCE[i], :, :] == color_linedt, axis=-1))[0]
        index_lane = np.mean(index_lane) if len(index_lane) > 0 else -1
        index_lanes.append(index_lane)
    index_lanes = np.array(index_lanes)
    mask = index_lanes != -1
    distance = convert_pixel_to_cm(index_lanes[mask] - arr_x_oriline[mask], ARRAY_WIDTH_LANE[mask], WIDTH_LANE) if len(mask) > 0 else []
    distance = np.mean(distance) if len(distance) > 0 else 0
    return distance

class Camera(object):
    _instance = None
    @staticmethod
    def get_instance():
        if Camera._instance is None:
            Camera._instance = Camera()
        return Camera._instance
    def __init__(self, car, cfg, status_control=False):
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

            self.status_control = status_control

            self.stream = cuda.Stream()
            self.car = car

            self.frame_recognized = None
            self.lines = None
            self.objects = None
            self.traffic_light_color = None
            self.thread_pool = ThreadPoolExecutor(max_workers=2)
            self.array_distance = cfg.ARRAY_DISTANCE
            self.array_width_lane = cfg.ARRAY_WIDTH_LANE
            self.width_lane = cfg.WIDTH_LANE
            self.center_servo = cfg.CENTER_SERVO
            self.min_duty_dc_run =cfg.MIN_DUTY_DC_RUN

            self.model_LD = LaneDetection(cfg.MODEL_LD_PATH, self.ori_height, self.ori_width, cfg.INPUT_HEIGHT_LD, cfg.INPUT_WIDTH_LD, cfg.CROP_RAITO, cfg.CLS_NUM_PER_LANE, 
                                          cfg.GRIDING_NUM, cfg.NUM_LANE, self.stream)
            self.model_YOLO = Yolo(cfg.MODEL_YOLO_PATH, self.ori_height, self.ori_width, cfg.INPUT_HEIGHT_YOLO, cfg.INPUT_WIDTH_YOLO, cfg.CLASSES, cfg.CONF_THRESH,
                                   cfg.NMS_THRESH, cfg.COLOR_DICT_HSV, self.stream)
            
            self.arr_x_oriline_right = []
            self.arr_x_oriline_left = []
            for i in range(len(self.array_distance)):
                width_lane = self.array_width_lane[i]
                x1 = self.center_width - width_lane // 2
                x2 = self.center_width + width_lane // 2
                self.arr_x_oriline_right.append(x2)
                self.arr_x_oriline_left.append(x1)
            self.arr_x_oriline_right = np.array(self.arr_x_oriline_right)
            self.arr_x_oriline_left = np.array(self.arr_x_oriline_left)

            self.time_turn_limmit = 3
            self.start_turn = None
            self.object_before = None
    def draw_line_limmit(self, frame_recognized):
        dis_step = 10 #cm
        # draw line
        for i in range(len(self.array_distance)):
            distance = self.array_distance[i]
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
        if frame_recognized is None:
            return
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

        self.frame_recognized, self.traffic_light_color = self.model_YOLO.get_objects_img(self.frame_recognized, self.objects)
        self.frame_recognized = draw_line_lane(self.frame_recognized, self.array_distance, self.array_width_lane, self.arr_x_oriline_left, self.arr_x_oriline_right)
        self.frame_recognized = self.model_LD.get_lane_img(self.frame_recognized, self.lines)

        if self.status_control:
            self.control_car()

        end = time.time()
        fps = 1 / (end - start)
        cv2.putText(self.frame_recognized, "FPS: {:.2f}".format(fps), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        return self.frame_recognized
    
    def control_car(self):
        try:
            self.control_car_meet_object()
            self.control_car_in_lane()
        except:
            pass
    
    def control_car_meet_object(self):
        if len(self.objects) > 0:
            dis_obj_limit = 3
            thre_width = 0.65
            speed = None
            # get object in range from 0cm to 30cm
            objects = self.objects[self.objects[:, 3] >= self.array_distance[dis_obj_limit]]
            
            if len(objects) > 0:
                dis_obj_limit -= 1
                mask_left = np.logical_and(objects[:,0] >= self.arr_x_oriline_left[dis_obj_limit], objects[:,0] <= self.arr_x_oriline_right[dis_obj_limit])
                mask_right = np.logical_and(objects[:,2] >= self.arr_x_oriline_left[dis_obj_limit], objects[:,2] <= self.arr_x_oriline_right[dis_obj_limit])
                object_min = None
                objects = objects[np.logical_or(mask_left, mask_right)]
                if len(objects) > 0:
                    # get object min distance
                    object_min = objects[np.argmax(objects[:, 3])]
                    # stop 20cm
                    if object_min[3] >= self.array_distance[1]:
                        self.car.stop()
                        raise Exception("return")
                if self.traffic_light_color:
                    if self.traffic_light_color == "stop":
                        self.car.stop()
                        raise Exception("return")
                    elif self.traffic_light_color == "warning":
                        self.car.speed = self.car.speed - 5
                        self.car.run()
                        raise Exception("return")
                    elif self.traffic_light_color == "go":
                        self.car.speed = self.min_duty_dc_run 
                
                # check object in lane
                if object_min[0] >= self.arr_x_oriline_left[dis_obj_limit] and object_min[2] <= self.arr_x_oriline_right[dis_obj_limit]:
                    self.start_turn = time.time()
                    return
                else:
                    dis_obj_left = np.abs(object_min[0] - self.arr_x_oriline_left[dis_obj_limit])
                    dis_obj_right = np.abs(self.arr_x_oriline_right[dis_obj_limit] - object_min[2])
                    # get max distance
                    if dis_obj_left > dis_obj_right:
                        if dis_obj_left >= thre_width * self.array_width_lane[dis_obj_limit]:
                            # turn left
                            dis = np.abs(object_min[0] - self.arr_x_oriline_right[dis_obj_limit])
                            dis = convert_pixel_to_cm(dis, self.array_width_lane[dis_obj_limit], self.width_lane)
                            self.car.turn_corner(self.center_servo - dis)
                            # if self.car.is_stop():
                            #     self.car.forward(self.min_duty_dc_run)
                        else:
                            # change lane
                            self.start_turn = time.time()
                            return
                    elif dis_obj_right >= thre_width * self.array_width_lane[dis_obj_limit]:
                        # turn left
                        dis = np.abs(object_min[2] - self.arr_x_oriline_left[dis_obj_limit])
                        dis = convert_pixel_to_cm(dis, self.array_width_lane[dis_obj_limit], self.width_lane)
                        self.car.turn_corner(self.center_servo + dis)
                        # if self.car.is_stop():
                        #     self.car.forward(self.min_duty_dc_run)
                    else:
                        self.car.stop()
                        return
                self.car.run()

    def control_car_in_lane(self):
        line_left = self.lines[0]
        line_right = self.lines[2]
        line_left_1 = self.lines[1]
        # get line from 10cm to 40cm
        index_dis = 3
        # line_left = line_left[line_left[:, 1] <= ARRAY_DISTANCE[0]]
        # line_left = line_left[line_left[:, 1] >= ARRAY_DISTANCE[index_dis]]
        # line_right = line_right[line_right[:, 1] <= ARRAY_DISTANCE[0]]
        # line_right = line_right[line_right[:, 1] >= ARRAY_DISTANCE[index_dis]]
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
        distance = get_distance_oriline_linedt(self.frame_recognized, self.array_distance, self.array_width_lane, self.width_lane, arr_x_oriline, colordt)
        thresh_dis = 2
        # smooth
        linedt_smooth = smooth_line(linedt)
        vector_oriline = np.array([arr_x_oriline[index_dis] - arr_x_oriline[0], self.array_distance[index_dis] - self.array_distance[0]])
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
        self.car.turn_corner(self.center_servo + angle_turn)
        self.car.run()

        