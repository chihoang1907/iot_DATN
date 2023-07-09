import tensorrt as trt
import cv2
import numpy as np
from utils_models.utils_Yolo import detect_color, post_processing
import pycuda.driver as cuda
import pycuda.autoinit
import time

class Yolo(object):
    _instance = None
    @staticmethod
    def get_instance():
        if Yolo._instance is None:
            Yolo._instance = Yolo()
        return Yolo._instance
    def __init__(self, model_path, ori_height, ori_width, input_height, input_width, classes, conf_th, nms_th, color_dict_HSV, stream):
        if Yolo._instance != None:
            raise Exception("This class is a singleton!")
        else:
            Yolo._instance = self
            self.model_path = model_path
            self.ori_height = ori_height
            self.ori_width = ori_width
            self.input_height = input_height
            self.input_width = input_width
            self.classes = classes
            self.num_classes = len(classes)
            self.conf_th = conf_th
            self.nms_th = nms_th
            self.color_dict_HSV = color_dict_HSV
            self.stream = stream
            self.init_model()
    def init_model(self):
        f = open(self.model_path, "rb")
        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        engine = runtime.deserialize_cuda_engine(f.read())
        self.context = engine.create_execution_context()
        self.output_loc = np.empty((1, 2535, 1, 4), dtype=np.float16)
        self.output_conf = np.empty((1, 2535, self.num_classes), dtype=np.float16)
        # Allocate device memory
        self.d_input = cuda.mem_alloc(1 * self.input_height * self.input_width * 3 * np.float16().nbytes)
        self.d_output_loc = cuda.mem_alloc(1 * self.output_loc.nbytes)
        self.d_output_conf = cuda.mem_alloc(1 * self.output_conf.nbytes)
        self.bindings = [int(self.d_input), int(self.d_output_loc), int(self.d_output_conf)]

    def predict(self, batch):
        # Copy to device
        cuda.memcpy_htod_async(self.d_input, batch, self.stream)
        # Run inference
        self.context.execute_async_v2(self.bindings, self.stream.handle, None)
        # Transfer predictions back
        cuda.memcpy_dtoh_async(self.output_loc, self.bindings[1], self.stream)
        cuda.memcpy_dtoh_async(self.output_conf, self.bindings[2], self.stream)
        # Synchronize the stream
        self.stream.synchronize()
        return self.output_loc, self.output_conf
    
    def preprocess(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.input_width, self.input_height))
        img = img.astype(np.float16)
        img = img / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)
        img = np.ascontiguousarray(img)
        return img
    
    def get_objects(self, output_loc, output_conf):
        bboxes_batch = post_processing(self.conf_th, self.nms_th, output_loc, output_conf)
        bboxes = bboxes_batch[0]
        if len(bboxes) > 0:
            bboxes[:, 0] *= self.ori_width
            bboxes[:, 1] *= self.ori_height
            bboxes[:, 2] *= self.ori_width
            bboxes[:, 3] *= self.ori_height
        return bboxes
    
    def get_objects_img(self, img, bboxes):
        traffic_light_color = None
        for box in bboxes:
            x1 = int(min(max(0, box[0]), self.ori_width - 1))
            y1 = int(min(max(0, box[1]), self.ori_height - 1))
            x2 = int(min(max(0, box[2]), self.ori_width - 1))
            y2 = int(min(max(0, box[3]), self.ori_height - 1))
            # w = x2 - x1
            # print(w, x2, y2)
            conf = box[4]
            cls_id = int(box[5])
            cls_name = self.classes[cls_id]
            if cls_name == "traffic light":
                traffic_light = img[y1:y2, x1:x2]
                color_name = detect_color(traffic_light, self.color_dict_HSV)
                traffic_light_color = color_name
                cv2.putText(img, color_name, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 1)
        return img, traffic_light_color
        