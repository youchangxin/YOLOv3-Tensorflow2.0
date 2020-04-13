# -*- coding: utf-8 -*-
import cv2
import numpy as np
import utils.tools as tools
import core.loss as loss
import tensorflow as tf
import time

from configuration import cfg
from utils.nms import nms
from core.models.yolov3 import YOLOv3


video_path      = "./data/road.mp4"
# video_path      = 0
num_classes     = 20
input_size      = 416

model = YOLOv3()
model.load_weights(filepath=cfg.YOLO.SAVE_MODEL_DIR + "saved_model")

vid = cv2.VideoCapture(video_path)
while True:
    return_value, frame = vid.read()
    if return_value:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    else:
        raise ValueError("No image!")
    frame_size = frame.shape[:2]
    image_data = tools.preprocess_data(np.copy(frame), [input_size, input_size])
    image_data = image_data[np.newaxis, ...].astype(np.float32)

    prev_time = time.time()

    feature_maps= model.predict(image_data)
    pred_bbox  = []
    for i, conv_tensor in enumerate(feature_maps):
        pred_tensor = loss.decode(conv_tensor, i)
        pred_bbox.append(pred_tensor)

    curr_time = time.time()
    exec_time = curr_time - prev_time

    pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]
    pred_bbox = tf.concat(pred_bbox, axis=0)
    bboxes = tools.postprocess_boxes(pred_bbox, frame_size, input_size, 0.3)
    bboxes = nms(bboxes, 0.45, method='nms')
    image = tools.draw_bbox(frame, bboxes)
    result = np.asarray(image)
    info = "time: %.2f ms" %(1000*exec_time)
    cv2.putText(result, text=info, org=(50, 70), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1, color=(255, 0, 0), thickness=2)
    cv2.namedWindow("result", cv2.WINDOW_AUTOSIZE)
    result = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imshow("result", result)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

