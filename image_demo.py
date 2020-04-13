# -*- coding: utf-8 -*-
import cv2
import numpy as np
import utils.tools as tools
import core.loss as loss
import tensorflow as tf

from configuration import cfg
from utils.nms import nms
from core.models.yolov3 import YOLOv3
from PIL import Image


input_size   = 416
image_path   = "./data/kite.jpg"
check_dir    = "./saved_model/"

original_image      = cv2.imread(image_path)
original_image      = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
original_image_size = original_image.shape[:2]

image_data = tools.preprocess_data(np.copy(original_image), [input_size, input_size])
image_data = image_data[np.newaxis, ...].astype(np.float32)

model = YOLOv3()
model.load_weights(filepath=cfg.YOLO.SAVE_MODEL_DIR + "saved_model")

feature_maps = model.predict(image_data)
decoded_tensor = []
for i, conv_tensor in enumerate(feature_maps):
    pred_tensor = loss.decode(conv_tensor, i)
    decoded_tensor.append(pred_tensor)

pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in decoded_tensor]
pred_bbox = tf.concat(pred_bbox, axis=0)

bboxes = tools.postprocess_boxes(pred_bbox, original_image_size, input_size, 0.3)
bboxes = nms(bboxes, 0.45, method='nms')

image = tools.draw_bbox(original_image, bboxes, show_label=True)
image = Image.fromarray(image)
image.show()


