# -*- coding: utf-8 -*-
import tensorflow as tf
from core.models.darknet53 import darknet53
import core.models.modules as modules
from configuration import cfg

NUM_CLASS       = len(cfg.YOLO.CLASSES)


class YOLOv3(tf.keras.Model):
    def __init__(self):
        super(YOLOv3, self).__init__()
        self.backbone = darknet53()
        self.conv_l1 = modules.Conv2d(filters=512, kernel_size=(1, 1))
        self.conv_l2 = modules.Conv2d(filters=1024, kernel_size=(3, 3))
        self.conv_l3 = modules.Conv2d(filters=512, kernel_size=(1, 1))
        self.conv_l4 = modules.Conv2d(filters=1024, kernel_size=(3, 3))
        self.conv_l5 = modules.Conv2d(filters=512, kernel_size=(1, 1))

        self.conv_large_branch = modules.Conv2d(filters=1024, kernel_size=(3, 3))
        self.conv_lbbox = tf.keras.layers.Conv2D(filters=3*(NUM_CLASS + 5), kernel_size=(1, 1),
                                                 strides=1, padding='same',
                                                 kernel_regularizer=tf.keras.regularizers.l2(0.0005),
                                                 kernel_initializer=tf.random_normal_initializer(stddev=0.01))

        #self.upsample_m = modules.
        self.conv_m = modules.Conv2d(filters=256, kernel_size=(1, 1))
        self.conv_m1 = modules.Conv2d(filters=256, kernel_size=(1, 1))
        self.conv_m2 = modules.Conv2d(filters=512, kernel_size=(3, 3))
        self.conv_m3 = modules.Conv2d(filters=256, kernel_size=(1, 1))
        self.conv_m4 = modules.Conv2d(filters=512, kernel_size=(3, 3))
        self.conv_m5 = modules.Conv2d(filters=256, kernel_size=(1, 1))
        self.conv_middle_branch = modules.Conv2d(filters=512, kernel_size=(3, 3))
        self.conv_mbbox = tf.keras.layers.Conv2D(filters=3 * (NUM_CLASS + 5), kernel_size=(1, 1),
                                                 strides=1, padding='same',
                                                 kernel_regularizer=tf.keras.regularizers.l2(0.0005),
                                                 kernel_initializer=tf.random_normal_initializer(stddev=0.01))

        self.conv_s = modules.Conv2d(filters=128, kernel_size=(1, 1))
        self.conv_s1 = modules.Conv2d(filters=128, kernel_size=(1, 1))
        self.conv_s2 = modules.Conv2d(filters=256, kernel_size=(3, 3))
        self.conv_s3 = modules.Conv2d(filters=128, kernel_size=(1, 1))
        self.conv_s4 = modules.Conv2d(filters=256, kernel_size=(3, 3))
        self.conv_s5 = modules.Conv2d(filters=128, kernel_size=(1, 1))
        self.conv_sobj_branch = modules.Conv2d(filters=256, kernel_size=(3, 3))
        self.conv_sbbox = tf.keras.layers.Conv2D(filters=3 * (NUM_CLASS + 5), kernel_size=(1, 1),
                                                 strides=1, padding='same',
                                                 kernel_regularizer=tf.keras.regularizers.l2(0.0005),
                                                 kernel_initializer=tf.random_normal_initializer(stddev=0.01))

    def call(self, inputs, training=None, mask=None):
        branch_1, branch_2, conv =self.backbone(inputs, training=training)

        # large object
        l = self.conv_l1(conv, training=training)
        l = self.conv_l2(l, training=training)
        l = self.conv_l3(l, training=training)
        l = self.conv_l4(l, training=training)
        l = self.conv_l5(l, training=training)
        conv_large_branch = self.conv_large_branch(l, training=training)
        conv_lbbox = self.conv_lbbox(conv_large_branch, training=training)

        # middle object
        m = self.conv_m(l, training=training)
        m = tf.image.resize(m, (m.shape[1] * 2, m.shape[2] * 2), method='nearest')
        m = tf.concat([m, branch_2], axis=-1)
        m = self.conv_m1(m, training=training)
        m = self.conv_m2(m, training=training)
        m = self.conv_m3(m, training=training)
        m = self.conv_m4(m, training=training)
        m = self.conv_m5(m, training=training)
        conv_middle_branch = self.conv_middle_branch(m, training=training)
        conv_mbbox = self.conv_mbbox(conv_middle_branch, training=training)

        # small object
        s = self.conv_s(m, training=training)
        s = tf.image.resize(s, (s.shape[1] * 2, s.shape[2] * 2), method='nearest')
        s = tf.concat([s, branch_1], axis=-1)
        s = self.conv_s1(s, training=training)
        s = self.conv_s2(s, training=training)
        s = self.conv_s3(s, training=training)
        s = self.conv_s4(s, training=training)
        s = self.conv_s5(s, training=training)
        conv_small_branch = self.conv_sobj_branch(s, training=training)
        conv_sbbox = self.conv_sbbox(conv_small_branch, training=training)

        return [conv_sbbox, conv_mbbox, conv_lbbox]


