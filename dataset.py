# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
from configuration import cfg
import utils.tools as tools
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'



class Dataset(object):
    def __init__(self, dataset_type):
        self.TFRECORD_DIR       = cfg.TRAIN.TFRECORD_DIR  if dataset_type == 'train' else cfg.TEST.TFRECORD_DIR
        self.input_sizes        = cfg.TRAIN.INPUT_SIZE if dataset_type == 'train' else cfg.TEST.INPUT_SIZE
        self.batch_size         = cfg.TRAIN.BATCH_SIZE if dataset_type == 'train' else cfg.TEST.BATCH_SIZE
        self.TXT_DIR            = cfg.TRAIN.TXT_DIR if dataset_type == 'train' else cfg.TEST.TXT_DIR

        self.strides            = np.array(cfg.YOLO.STRIDES)
        self.anchor_per_scale   = cfg.YOLO.ANCHOR_PER_SCALE
        self.data_aug           = cfg.TRAIN.DATA_AUG
        self.output_sizes       = [self.input_sizes] // self.strides
        self.classes            = cfg.YOLO.CLASSES
        self.num_classes        = len(self.classes)
        self.max_bbox_per_scale = cfg.YOLO.MAX_PER_SCALE

        self.anchors            = np.array(tools.get_anchors(cfg.YOLO.ANCHORS))
        self.num_samples        = tools.get_num_annatations(self.TXT_DIR)
        self.num_batchs         = int(np.ceil(self.num_samples / self.batch_size))


    def __parse_example(self, example_string):
        # define structure of Feature
        feature_description = {
            'image': tf.io.VarLenFeature(dtype=tf.string),
            'image_height': tf.io.FixedLenFeature([], tf.int64),
            'image_width': tf.io.FixedLenFeature([], tf.int64),
            'boxes': tf.io.VarLenFeature(dtype=tf.int64),
        }
        # decode the TFRecord file
        feature_dict = tf.io.parse_single_example(example_string, feature_description)

        # transite SparseTensor to DenseTensor
        feature_dict['image'] = tf.sparse.to_dense(feature_dict['image'])
        feature_dict['boxes'] = tf.sparse.to_dense(feature_dict['boxes'])
        # reshape tensor
        feature_dict['image'] = tf.reshape(feature_dict['image'], [])
        feature_dict['image_height'] = tf.reshape(feature_dict['image_height'], [])
        feature_dict['image_width'] = tf.reshape(feature_dict['image_width'], [])
        feature_dict['boxes'] = tf.reshape(feature_dict['boxes'], [-1, 5])

        h, w = feature_dict['image_height'], feature_dict['image_width']

        feature_dict['image'] = tf.io.decode_jpeg(feature_dict['image'], channels=3)
        feature_dict['image'] = tf.image.resize(feature_dict['image'], [h, w])

        return feature_dict['image'], feature_dict['boxes']


    def __generate_label(self, image, bboxes):
        [image, batch_label_sbbox, batch_label_mbbox , batch_label_lbbox, batch_sbboxes, mbboxes, lbboxes] = tf.py_function(func=self.__generate_label_py_func,
                                        inp=[image, bboxes],
                                        Tout=[tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32])
        return image, batch_label_sbbox, batch_label_mbbox , batch_label_lbbox, batch_sbboxes, mbboxes, lbboxes


    def __data_process(self, image, boxes):
        [image, boxes] = tf.py_function(func=tools.preprocess_data,
                                        inp=[image, [self.input_sizes, self.input_sizes], boxes, self.data_aug],
                                        Tout=[tf.float32, tf.int32])
        return image, boxes


    def __generate_label_py_func(self, image, bboxes):

        # 对不同尺度的feature map 产生3个default boxes，与GT进行iou运算产生label bboxes
        batch_size = image.shape[0]
        batch_label_sbbox = np.zeros((batch_size, self.output_sizes[0], self.output_sizes[0],
                                      self.anchor_per_scale,5 + self.num_classes), dtype=np.float32)
        batch_label_mbbox = np.zeros((batch_size, self.output_sizes[1], self.output_sizes[1],
                                      self.anchor_per_scale, 5 + self.num_classes), dtype=np.float32)
        batch_label_lbbox = np.zeros((batch_size, self.output_sizes[2], self.output_sizes[2],
                                      self.anchor_per_scale, 5 + self.num_classes), dtype=np.float32)

        # 保存不同尺寸的feature map下的每张图片的bbox
        batch_sbboxes = np.zeros((batch_size, self.max_bbox_per_scale, 4), dtype=np.float32)
        batch_mbboxes = np.zeros((batch_size, self.max_bbox_per_scale, 4), dtype=np.float32)
        batch_lbboxes = np.zeros((batch_size, self.max_bbox_per_scale, 4), dtype=np.float32)

        bboxes = np.array(bboxes)

        for num in range(tools.get_valid_num_bboxes(bboxes)):
            label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes = self.preprocess_true_boxes(bboxes[num])

            batch_label_sbbox[num, :, :, :, :] = label_sbbox
            batch_label_mbbox[num, :, :, :, :] = label_mbbox
            batch_label_lbbox[num, :, :, :, :] = label_lbbox
            batch_sbboxes[num, :, :] = sbboxes
            batch_mbboxes[num, :, :] = mbboxes
            batch_lbboxes[num, :, :] = lbboxes

        return image, batch_label_sbbox, batch_label_mbbox , batch_label_lbbox, batch_sbboxes, batch_mbboxes, batch_mbboxes


    def preprocess_true_boxes(self, bboxes):
        # 把原图像分别做8,16,32倍下采样变化。假设原始输入为416,416  3次下采样为
        # [(52,52,3,5+20),(26,26,3,5+20),(13,13,3,5+20)]
        label = [np.zeros((self.output_sizes[i], self.output_sizes[i], self.anchor_per_scale,
                           5 + self.num_classes)) for i in range(3)]

        # [(150,4),(150,4),(150,4)]，存储实际得box，最多每张图片允许存在150个box
        bboxes_xywh = [np.zeros((self.max_bbox_per_scale, 4)) for _ in range(3)]
        # 对每个下采样图像得box数目进行计数
        bbox_count = np.zeros(shape=(3, ))

        for bbox in bboxes:
            bbox_coordinate = bbox[:4]
            bbox_class_ind = bbox[4]

            # convet to one_hot code
            onehot = np.zeros(self.num_classes, dtype=np.float)
            onehot[bbox_class_ind] = 1.0
            # smooth label
            uniform_distribution = np.full(self.num_classes, 1.0 / self.num_classes)
            deta = 0.01
            smooth_onehot = onehot * (1-deta) + deta * uniform_distribution

            # conver to xywh
            bbox_xywh = np.concatenate([(bbox_coordinate[2:] + bbox_coordinate[:2]) * 0.5,
                                         bbox_coordinate[2:] - bbox_coordinate[:2]], axis=-1) #shape[1, 3]
            bbox_xywh_scale = 1.0 * bbox_xywh[np.newaxis, :] /self.strides[:, np.newaxis]  # shape[3,4]

            iou = []
            exist_postive = False
            for i in range(3): # loop in 3 scale of feature map
                anchors_xywh = np.zeros((self.anchor_per_scale, 4)) #shape(3, 4)
                anchors_xywh[:, 0:2] = np.floor(bbox_xywh_scale[i, 0:2]).astype(np.int32) + 0.5
                # 赋值对应的anchors，这里为"./data/anchors/basline_anchors.txt"中
                # 1.25,  1.625,     2.0,   3.75,       4.125,    2.875,         i=0   8倍下采样
                # 1.875, 3.8125,    3.875, 2.8125,     3.6875,   7.4375,        i=1   16倍下采样
                # 3.625, 2.8125,    4.875, 6.1875,     11.65625, 10.1875        i=2   32倍下采样
                anchors_xywh[:, 2:4] = self.anchors[i]
                #                                   (1, 4)                     (3, 4)
                iou_scale = tools.bbox_iou(bbox_xywh_scale[i][np.newaxis, :], anchors_xywh)
                iou.append(iou_scale)
                iou_mask = iou_scale > 0.3

                if np.any(iou_mask):    # 3个default box任意一个iou超过0.3表示其为一个正样本，然后进行处理
                    x_centre, y_centre = np.floor(bbox_xywh_scale[i, 0:2]).astype(np.int32)
                    label[i][y_centre, x_centre, iou_mask, :] = 0
                    label[i][y_centre, x_centre, iou_mask, 0:4] = bbox_xywh
                    label[i][y_centre, x_centre, iou_mask, 4:5] = 1.0
                    label[i][y_centre, x_centre, iou_mask, 5:] = smooth_onehot

                    bbox_ind = int(bbox_count[i] % self.max_bbox_per_scale)
                    bboxes_xywh[i][bbox_ind, :4] = bbox_xywh
                    bbox_count[i] += 1

                    exist_postive = True

            if not exist_postive: # bbox没有正样本
                best_anchor_ind = np.argmax(np.array(iou).reshape(-1), axis=-1) # 行方向求最大值的index

                best_scale = int(best_anchor_ind / self.anchor_per_scale)
                best_anchor = int(best_anchor_ind % self.anchor_per_scale)

                x_centre, y_centre = np.floor(bbox_xywh_scale[best_scale, 0:2]).astype(np.int32)

                label[best_scale][y_centre, x_centre, best_scale, :] = 0
                label[best_scale][y_centre, x_centre, best_scale, 0:4] = bbox_xywh
                label[best_scale][y_centre, x_centre, best_scale, 4:5] = 1.0
                label[best_scale][y_centre, x_centre, best_scale, 5:] = smooth_onehot

                bbox_ind = int(bbox_count[best_scale] % self.max_bbox_per_scale)
                bboxes_xywh[best_scale][bbox_ind, :4] = bbox_xywh
                bbox_count[best_scale] += 1

        label_sbbox, label_mbbox, label_lbbox = label
        sbboxes, mbboxes, lbboxes = bboxes_xywh

        return label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes


    def _read_annatations(self):
        dataset = tf.data.TFRecordDataset(self.TFRECORD_DIR)
        dataset = dataset.map(map_func=self.__parse_example,
                              num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.map(map_func=self.__data_process,
                              num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.shuffle(buffer_size=5000)
        dataset = dataset.batch(batch_size=self.batch_size)
        dataset = dataset.map(map_func=self.__generate_label,
                              num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        return  dataset


    def __len__(self):
        return self.num_batchs

  #'''
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    dataset = Dataset('train')._read_annatations()
    for image, batch_label_sbbox, batch_label_mbbox , batch_label_lbbox, batch_sbboxes, mbboxes, lbboxes in dataset:
        print(mbboxes)


        print(image.shape)
        image = tf.squeeze(image)
        image = tf.cast(image, tf.int32)

        plt.imshow(image.numpy())
        plt.show()
# '''


