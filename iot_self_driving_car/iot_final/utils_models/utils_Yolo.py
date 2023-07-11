import cv2
import numpy as np

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

def detect_color(traffic_light, color_dict_HSV):
    traffic_light = cv2.cvtColor(traffic_light, cv2.COLOR_BGR2HSV)
    upper_red, lower_red = color_dict_HSV['red_1']
    red_mask_1 = cv2.inRange(traffic_light, lower_red, upper_red)
    upper_red, lower_red = color_dict_HSV['red_2']
    red_mask_2 = cv2.inRange(traffic_light, lower_red, upper_red)
    red_mask = red_mask_1 + red_mask_2
    upper_go, lower_go = color_dict_HSV['go']
    go_mask = cv2.inRange(traffic_light, lower_go, upper_go)
    upper_warning, lower_warning = color_dict_HSV['warning']
    warning_mask = cv2.inRange(traffic_light, lower_warning, upper_warning)
    red_mask = np.sum(red_mask == 255)
    go_mask = np.sum(go_mask == 255)
    warning_mask = np.sum(warning_mask == 255)
    color_mask = None
    color_name = None
    if red_mask > go_mask and red_mask > warning_mask:
        color_mask = red_mask
        color_name = 'stop'
    elif go_mask > red_mask and go_mask > warning_mask:
        color_mask = go_mask
        color_name = 'go'
    elif warning_mask > red_mask and warning_mask > go_mask:
        color_mask = warning_mask
        color_name = 'warning'
    if color_mask and color_mask >= 100:
        return color_name
    else:
        return None