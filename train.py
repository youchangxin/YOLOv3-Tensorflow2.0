# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import time
import os
import shutil
import core.loss as loss

from configuration import cfg
from tqdm import tqdm
from dataset import Dataset
from core.loss import decode
from core.models.yolov3 import YOLOv3

#os.environ["PATH"] += os.pathsep + r'D:\Program Files\Graphviz2.38\bin'  #注意修改你的路径


logdir = "./log"
save_frequency = cfg.TRAIN.SAVE_FREQ

# GPU Setting
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


global_steps = tf.Variable(initial_value=1, trainable=False, dtype=tf.int64)

trainset = Dataset('train')
steps_per_epoch = len(trainset)
warmup_steps = cfg.TRAIN.WARMUP_EPOCHS * steps_per_epoch
total_steps = cfg.TRAIN.EPOCHS * steps_per_epoch

# model
model = YOLOv3()
model.build(input_shape=(None, cfg.TRAIN.INPUT_SIZE, cfg.TRAIN.INPUT_SIZE, 3))
model.summary()

# TensorBoard
if os.path.exists(logdir): shutil.rmtree(logdir)
writer = tf.summary.create_file_writer(logdir)

optimizer = tf.keras.optimizers.Adam()

def train_step(images, target):
    with tf.GradientTape() as tape:
        pred_results = model(images, training=True)

        decoded_tensor = []
        for i, conv_tensor in enumerate(pred_results):
            pred_tensor = loss.decode(conv_tensor, i)
            decoded_tensor.append(conv_tensor)
            decoded_tensor.append(pred_tensor)

        # Computing LOSS
        giou_loss = conf_loss = prob_loss = 0
        for i in range(3):
            conv, pred = decoded_tensor[i * 2], decoded_tensor[i * 2 + 1]
            # *用于参数前面，表示传入的多个参数将按照元组的形式存储，是一个元组；
            loss_items = loss.yolov3_loss(pred, conv, *target[i], i)
            giou_loss += loss_items[0]
            conf_loss += loss_items[1]
            prob_loss += loss_items[2]

        total_loss = giou_loss + conf_loss + prob_loss

        gradients = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        tf.print("=> STEP %4d  lr: %.6f  giou_loss: %4.2f  conf_loss: %4.2f  "
                 "prob_loss: %4.2f  total_loss: %4.2f" %(global_steps, optimizer.lr.numpy(),
                                                          giou_loss, conf_loss,
                                                          prob_loss, total_loss))

        # update learning rate
        global_steps.assign_add(1)
        if global_steps < warmup_steps:
            lr = global_steps / warmup_steps * cfg.TRAIN.LR_INIT
        else:
            lr = cfg.TRAIN.LR_END + 0.5 * (cfg.TRAIN.LR_INIT - cfg.TRAIN.LR_END) *(
                    (1 + tf.cos((global_steps - warmup_steps) / (total_steps - warmup_steps) * np.pi))
                 )
        optimizer.lr.assign(lr.numpy())

        # writing summary data
        with writer.as_default():
            tf.summary.scalar("lr", optimizer.lr, step=global_steps)
            tf.summary.scalar("loss/total_loss", total_loss, step=global_steps)
            tf.summary.scalar("loss/giou_loss", giou_loss, step=global_steps)
            tf.summary.scalar("loss/conf_loss", conf_loss, step=global_steps)
            tf.summary.scalar("loss/prob_loss", prob_loss, step=global_steps)
        writer.flush()

if __name__ == "__main__":

    for epoch in range(cfg.TRAIN.EPOCHS):

        tf.print('=> EPOCH:  %3d / %3d  ' % (epoch, cfg.TRAIN.EPOCHS))

        for batch_image, batch_label_sbbox, batch_label_mbbox , batch_label_lbbox, batch_sbboxes, batch_mbboxes, batch_lbboxes in trainset._read_annatations():

            batch_small_target = batch_label_sbbox, batch_sbboxes
            batch_medium_target = batch_label_mbbox, batch_mbboxes
            batch_larger_target = batch_label_lbbox, batch_lbboxes
            target = (batch_small_target, batch_medium_target, batch_larger_target)

            train_step(batch_image, target)

        # save model weights
        if epoch % save_frequency == 0:
            model.save_weights(filepath=cfg.YOLO.SAVE_MODEL_DIR + "YOLOv3_epoch-{}".format(epoch), save_format="tf")

    model.save_weights(filepath=cfg.YOLO.SAVE_MODEL_DIR + "saved_model", save_format="tf")






