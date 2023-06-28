import tensorrt as trt
import cv2
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
import time

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
        bboxes_batch = np.array(bboxes_batch)
    return bboxes_batch

class Yolo(object):
    _instance = None
    @staticmethod
    def get_instance():
        if Yolo._instance is None:
            Yolo._instance = Yolo()
        return Yolo._instance
    def __init__(self, model_path, ori_height, ori_width, input_height, input_width, classes, conf_th, nms_th, stream):
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
        for box in bboxes:
            x1 = int(box[0])
            y1 = int(box[1])
            x2 = int(box[2])
            y2 = int(box[3])
            # w = x2 - x1
            # print(w, x2, y2)
            conf = box[4]
            cls_id = int(box[5])
            cls_name = self.classes[cls_id]
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 1)
            # cv2.putText(img, cls_name + " " + str(np.round(conf, 2)), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        return img
        