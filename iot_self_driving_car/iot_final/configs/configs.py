import numpy as np

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

MODEL_LD_PATH = "export_models/ufld_mydataset_2_engine.trt"

#YOLO
CLASSES = [
    "object",
    "traffic light"
]
NUM_CLASS = len(CLASSES)
INPUT_HEIGHT_YOLO = 416
INPUT_WIDTH_YOLO = 416
MODEL_YOLO_PATH = "export_models/yolov4_my_dataset_engine.trt"
NMS_THRESH = 0.4
CONF_THRESH = 0.8
CENTER_SERVO = 95

COLOR_DICT_HSV = {
        'white': [[174, 255, 230], [0, 4, 140]],
        'go': [[92, 255, 255], [40, 0, 200]],
        'red_1': [[180, 255, 255], [165, 15, 220]],
        'red_2': [[8, 255, 255], [0, 43, 220]],
        'warning': [[130, 255, 255], [85, 19, 200]],
}
for key in COLOR_DICT_HSV:
    COLOR_DICT_HSV[key] = np.array(COLOR_DICT_HSV[key])


# Car
MIN_DUTY_DC_RUN = 50
TIME_SLEEP_DC_RUN = 0.1
MIN_SERVO = 65
MAX_SERVO = 135
CENTER_SERVO = 95
ARRAY_DISTANCE = np.array([454, 376, 331, 292, 266]) # 10, 20, 30, 40 cm
ARRAY_WIDTH_LANE = np.array([546, 442, 378, 320, 288]) # 10, 20, 30, 40 cm
WIDTH_LANE=30 #cm