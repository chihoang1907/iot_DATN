import tensorrt as trt
import cv2
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
import time

class LaneDetection(object):
    _instance = None
    @staticmethod
    def get_instance():
        if LaneDetection._instance is None:
            LaneDetection._instance = LaneDetection()
        return LaneDetection._instance
    def __init__(self, model_path, ori_height, ori_width, input_height, input_width, crop_raito, cls_num_per_lane, griding_num, num_lane, stream):
        if LaneDetection._instance != None:
            raise Exception("This class is a singleton!")
        else:
            LaneDetection._instance = self
            self.model_path = model_path
            self.ori_height = ori_height
            self.ori_width = ori_width
            self.input_height = input_height
            self.input_width = input_width
            self.cls_num_per_lane = cls_num_per_lane
            self.num_lane = num_lane
            self.crop_raito = crop_raito
            self.griding_num = griding_num
            self.row_anchor = np.linspace(self.input_height*self.crop_raito, self.input_height-1, self.cls_num_per_lane).astype(int)
            self.diff = self.row_anchor[1:] - self.row_anchor[:-1]
            self.diff = np.mean(self.diff)
            self.col_sample = np.linspace(0, self.input_width - 1, self.griding_num)
            self.col_sample_w = self.col_sample[1] - self.col_sample[0]
            self.stream = stream
            self.init_model()

    
    def init_model(self):
        f = open(self.model_path, "rb")
        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        engine = runtime.deserialize_cuda_engine(f.read())
        context = engine.create_execution_context()
        self.context = context
        self.output = np.empty((1, self.num_lane, self.cls_num_per_lane, 2), dtype=np.float16)
        self.d_input = cuda.mem_alloc(1 * 3 * self.input_height * self.input_width * np.dtype(np.float16).itemsize)
        self.d_output = cuda.mem_alloc(1 * self.output.nbytes)
        self.bindings = [int(self.d_input), int(self.d_output)]
    def __del__(self):
        pass
    def predict(self, batch): # result gets copied into output
        # Transfer input data to device
        cuda.memcpy_htod_async(self.d_input, batch, self.stream)
        # Execute model
        self.context.execute_async_v2(self.bindings, self.stream.handle, None)
        # Transfer predictions back
        cuda.memcpy_dtoh_async(self.output, self.d_output, self.stream)
        # Syncronize threads
        self.stream.synchronize()
        return self.output
    def preprocess(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.input_width, self.input_height))
        img = np.array(img)
        img = img / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0).astype(np.float16)
        img = np.ascontiguousarray(img)
        return img
    def get_lane(self, img):
        out = self.predict(img)
        lines = out[0]
        lines_list = []
        for line in lines:
            # filter out location =0
            line = line[line[:, 0] != 0]
            if line.shape[0] == 0:
                lines_list.append(line)
                continue
            line[:,0] = ((line[:, 0] * self.col_sample_w * self.ori_width / self.input_width) - 1).astype(int)
            line[:,1] = (self.ori_height - line[:, 1] * self.diff * self.ori_height / self.input_height).astype(int)
            lines_list.append(line)
        return lines_list
    def get_lane_img(self, img, lines_list):
        colors = [(255, 0, 0), (255, 255, 0), (0, 0, 255), (255,255,0)]
        for i in range(len(lines_list)):
            line = lines_list[i]
            # draw line
            for j in range(line.shape[0] - 1):
                p1 = (line[j, 0], line[j, 1])
                p2 = (line[j + 1, 0], line[j + 1, 1])
                cv2.line(img, p1, p2, colors[i], 1)
        return img