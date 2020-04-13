# -*- coding: utf-8 -*-
import tensorflow as tf
import core.models.modules as modules


def build_resnet_block(n, filter1, filter2):
    block = tf.keras.Sequential()
    for _ in range(n):
        block.add(modules.residual_block(filter1, filter2))
    return block

class darknet53(tf.keras.layers.Layer):
    def __init__(self):
        super(darknet53, self).__init__()
        self.conv1 = modules.Conv2d(filters=32, kernel_size=(3, 3))
        self.conv2_d = modules.Conv2d_Downsample(filters=64, kernel_size=(3, 3))
        self.res1 = build_resnet_block(1, 32, 64)
        self.conv3_d = modules.Conv2d_Downsample(filters=128, kernel_size=(3, 3))
        self.res2 = build_resnet_block(2, 64, 128)
        self.conv4_d = modules.Conv2d_Downsample(filters=256, kernel_size=(3, 3))
        self.res3 = build_resnet_block(8, 128, 256)
        self.conv5_d = modules.Conv2d_Downsample(filters=512, kernel_size=(3, 3))
        self.res4 = build_resnet_block(8, 256, 512)
        self.conv6_d = modules.Conv2d_Downsample(filters=1024, kernel_size=(3, 3))
        self.res5 = build_resnet_block(4, 512, 1024)

    def call(self, inputs, training = None, **kwargs):
        x = self.conv1(inputs,training=training)
        x = self.conv2_d(x, training=training)
        x = self.res1(x, training=training)
        x = self.conv3_d(x, training=training)
        x = self.res2(x, training=training)
        x = self.conv4_d(x, training=training)
        x = self.res3(x, training=training)
        branch_1 = x
        x = self.conv5_d(x, training=training)
        x = self.res4(x, training=training)
        branch_2 = x
        x = self.conv6_d(x, training=training)
        x = self.res5(x, training=training)

        return branch_1, branch_2, x






