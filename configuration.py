# -*- coding: utf-8 -*-
from easydict import EasyDict as edict

__C                             = edict()
# Consumers can get config by: from config import cfg
cfg                             = __C

# YOLO section
__C.YOLO                        = edict()

__C.YOLO.STRIDES                = [8, 16, 32]
__C.YOLO.ANCHOR_PER_SCALE       = 3
__C.YOLO.MAX_PER_IMAGE          = 450
__C.YOLO.MAX_PER_SCALE          = 150
__C.YOLO.CLASSES                = { "person"     :  0, "bird"       :  1, "cat"    :  2, "cow"       :  3,
                                    "dog"        :  4, "horse"      :  5, "sheep"  :  6, "aeroplane" :  7,
                                    "bicycle"    :  8, "boat"       :  9, "bus"    : 10, "car"       : 11,
                                    "motorbike"  : 12, "train"      : 13, "bottle" : 14, "chair"     : 15,
                                    "diningtable": 16, "pottedplant": 17, "sofa"   : 18, "tvmonitor" : 19 }

__C.YOLO.ANCHORS                = "./data/anchors/basline_anchors.txt"
__C.YOLO.SAVE_MODEL_DIR         = "./saved_model/"
__C.YOLO.IOU_LOSS_THRES         = 0.5



# Train section
__C.TRAIN                       = edict()
__C.TRAIN.TFRECORD_DIR          = "data/train.tfrecord"
__C.TRAIN.TXT_DIR               = "data/dataset.txt"
__C.TRAIN.BATCH_SIZE            = 8
__C.TRAIN.INPUT_SIZE            = 416
__C.TRAIN.DATA_AUG              = True
__C.TRAIN.LR_INIT               = 1e-3
__C.TRAIN.LR_END                = 1e-6
__C.TRAIN.WARMUP_EPOCHS         = 2
__C.TRAIN.EPOCHS                = 20
__C.TRAIN.SAVE_FREQ             = 5

# Test section
__C.TEST                        = edict()
__C.TEST.TFRECORD_DIR          = "data/train.tfrecord"
__C.TEST.BATCH_SIZE             = 2
__C.TEST.INPUT_SIZE             = 544
__C.TEST.DECTECTED_IMAGE_PATH   = "./data/detection/"
__C.TEST.SCORE_THRESHOLD        = 0.3
__C.TEST.IOU_THRESHOLD          = 0.45