import tensorrt as trt
from scipy.special import softmax
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

def post_processing_yolo(conf_thresh, nms_thresh, box_array, confs, ori_height, ori_width):
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

            if (len(keep) > 0):
                ll_box_array = ll_box_array[keep, :]
                ll_max_conf = ll_max_conf[keep]
                ll_max_id = ll_max_id[keep]

                for k in range(ll_box_array.shape[0]):
                    bboxes.append([ll_box_array[k, 0]*ori_width, ll_box_array[k, 1]*ori_height, ll_box_array[k, 2]*ori_width, ll_box_array[k, 3]*ori_height,
                                    ll_max_conf[k], ll_max_id[k]])

        bboxes_batch.append(bboxes)
    return bboxes_batch

def preprocess_yolo(img, input_size):
    img = cv2.resize(img, (input_size, input_size), interpolation=cv2.INTER_LINEAR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float16)
    img = img / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    img = np.ascontiguousarray(img)
    return img

def forward_yolo(img, ori_height, ori_width, fn_predict):    
    output_loc, output_conf = fn_predict(img)
    bbox_batch = post_processing_yolo(0.4, 0.6, output_loc, output_conf, ori_height, ori_width)
    return bbox_batch

def generate_lines(out, griding_num, input_height, input_width, ori_height, ori_width, diff):

    col_sample = np.linspace(0, input_width - 1, griding_num)
    col_sample_w = col_sample[1] - col_sample[0]

    lines_batch = []
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
        lines = []
        for i in range(out_j.shape[1]):
            line = []
            if np.sum(out_j[:, i] != 0) > 2:
                for k in range(out_j.shape[0]):
                    if out_j[k, i] > 0:
                        line.append(int(out_j[k, i] * col_sample_w * ori_width / input_width) - 1)
                        line.append(int(ori_height - k * diff * ori_height / input_height) - 1)
            lines.append(line)
        lines_batch.append(lines)
    return lines_batch
    
def show_img_with_boxes(img, boxes, classes):
    for box in boxes:
        cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255,0,0), 2)
        cv2.putText(img, classes[int(box[5])], (int(box[0]), int(box[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    return img

def show_img_with_lines(img, lines):
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255,255,0)]
    for i in range(len(lines)):
        line = lines[i]
        for p in range(0, len(line), 2):
            cv2.circle(img, (line[p], line[p+1]), 5, colors[i], -1)
    return img

def preprocess_ufld(img, input_height, input_width):
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
    return img

def foward_ufld(img, ori_height, ori_width, input_height, input_width, griding_num, diff, fn_predict):
    out = fn_predict(img)
    lines_batch = generate_lines(out, griding_num, input_height, input_width, ori_height, ori_width, diff)
    return lines_batch