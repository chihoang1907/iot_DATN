import tensorflow as tf
import cv2
import numpy as np
import time

video_test_path = "example.mp4"
video_save_path = "example_out_ssd.mp4"

NUM_CLASS = 5

CLASSES = ['traffic light',
 'car',
 'person',
 'bus',
 'truck',
]

print("Start processing...")
# load the model
saved_model = tf.saved_model.load('saved_model_trt_fp16')
model = saved_model.signatures["serving_default"]

# load the video
cap = cv2.VideoCapture(video_test_path)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out_video = cv2.VideoWriter(video_save_path, fourcc, fps, (width, height))

print("Start processing...")
while True:
    start = time.time()
    ret, frame = cap.read()
    if not ret:
        break
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (320, 320))
    img = img.astype(np.float32)
    img = np.expand_dims(img, axis=0)


    # run the model
    output_dict = model(tf.constant(img))

    detection_boxes = output_dict['detection_boxes'][0].numpy()
    detection_scores = output_dict['detection_scores'][0].numpy()
    detection_classes = output_dict['detection_classes'][0].numpy().astype(np.int32)

    # draw the bounding boxes
    for i in range(len(detection_boxes)):
        if detection_scores[i] > 0.4:
            ymin = int(detection_boxes[i][0] * height)
            xmin = int(detection_boxes[i][1] * width)
            ymax = int(detection_boxes[i][2] * height)
            xmax = int(detection_boxes[i][3] * width)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            cv2.putText(frame, CLASSES[detection_classes[i]-1], (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    fps = 1.0 / (time.time() - start)
    cv2.putText(frame, "FPS: %.2f" % fps, (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    out_video.write(frame)
