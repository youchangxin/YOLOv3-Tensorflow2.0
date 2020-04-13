# -*- coding: utf-8 -*-
import tensorflow as tf

'''
class BatchNormalization(tf.keras.layers.BatchNormalization):
    """
    "Frozen state" and "inference mode" are two separate concepts.
    `layer.trainable = False` is to freeze the layer, so the layer will use
    stored moving `var` and `mean` in the "inference mode", and both `gama`
    and `beta` will not be updated !
    """
    def call(self, inputs, training=False):
        if not training:
            training = tf.constant(False)
        training = tf.logical_and(training, self.trainable)
        return super().call(inputs, training)
'''


class Conv2d(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size):
        super(Conv2d, self).__init__()
        self.conv = tf.keras.layers.Conv2D(filters=filters,
                                           kernel_size=kernel_size,
                                           strides=1,
                                           kernel_regularizer=tf.keras.regularizers.l2(0.0005),
                                           kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                           padding='same')
        self.bn = tf.keras.layers.BatchNormalization()

    def call(self, inputs, training=None, **kwargs):
        x = self.conv(inputs, training=training)
        x = self.bn(x, training=training)
        x = tf.nn.leaky_relu(x)

        return x


class Conv2d_Downsample(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size):
        super(Conv2d_Downsample, self).__init__()
        self.zp = tf.keras.layers.ZeroPadding2D(padding=((1, 0), (1, 0)))
        self.conv = tf.keras.layers.Conv2D(filters=filters,
                                           kernel_size=kernel_size,
                                           strides=2,
                                           kernel_regularizer=tf.keras.regularizers.l2(0.0005),
                                           kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                           padding='valid')
        self.bn = tf.keras.layers.BatchNormalization()

    def call(self, inputs, training=None, **kwargs):
        x = self.zp(inputs)
        x = self.conv(x)
        x = self.bn(x, training=training)
        x = tf.nn.leaky_relu(x)

        return x


class residual_block(tf.keras.layers.Layer):
    def __init__(self, filter1, filter2):
        super(residual_block, self).__init__()
        self.conv1 = Conv2d(filters=filter1, kernel_size=(1, 1))
        self.conv2 = Conv2d(filters=filter2, kernel_size=(3, 3))

    def call(self, inputs, training=None, **kwargs):
        x = self.conv1(inputs, training=training)
        x = self.conv2(x, training=training)

        short_cut = inputs + x
        return short_cut
