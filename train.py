import tensorflow as tf
import numpy as np
from datetime import datetime
from sys import stdout
import shutil
import os

from data_reader import AsyncReader, ReaderOpts
from loss import LossLayer
from model import build_flow_net
from drawing import display_training

# TODO: Data augmentation

img_height = 192
img_width = 640
# img_height = 96
# img_width = 320

kitti_path = '/home/xian/kitti_data'
batch_size = 4
nworkers = 6
train_reader_opts = ReaderOpts(kitti_path, batch_size, img_height, img_width, nworkers)

nepochs = 20

pretrained_weights_path = '/home/xian/ckpts/resnet18_fully_trained/ckpt'

flow_net = build_flow_net(img_height, img_width, pretrained_weights_path)
flow_net.summary()

trainable_weights = flow_net.trainable_weights

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

loss_layer = LossLayer()

log_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'logs')
if os.path.exists(log_dir):
    shutil.rmtree(log_dir)
train_summary_writer = tf.summary.create_file_writer(os.path.join(log_dir, 'train'))
val_summary_writer = tf.summary.create_file_writer(os.path.join(log_dir, 'val'))

save_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'ckpts')


@tf.function
def train_step(batch_imgs, step_count):
    # batch_imgs: (batch_size, height, width, 6)
    with tf.GradientTape() as tape:
        flows = flow_net(batch_imgs)
        # TODO: I can spare one of this concatenations, but for now this is clearer:
        loss_value = loss_layer(batch_imgs, flows)

    grads = tape.gradient(loss_value, trainable_weights)
    optimizer.apply_gradients(zip(grads, trainable_weights))

    with train_summary_writer.as_default():
        tf.summary.scalar('loss', loss_value, step=step_count)

    return loss_value, flows


with AsyncReader(train_reader_opts) as train_reader:
    step_count = 0
    for epoch in range(nepochs):
        print("\nStart epoch ", epoch + 1)
        epoch_start = datetime.now()
        if epoch == 15:
            optimizer.learning_rate = 1e-5
            print('Changing learning rate to: %.2e' % optimizer.learning_rate)
        for batch_idx in range(train_reader.nbatches):
            batch_imgs = train_reader.get_batch()
            step_count_tf = tf.convert_to_tensor(step_count, dtype=tf.int64)
            batch_imgs_tf = tf.convert_to_tensor(batch_imgs, dtype=tf.float32)
            loss_value, flows = train_step(batch_imgs_tf, step_count_tf)
            train_summary_writer.flush()
            stdout.write("\rbatch %d/%d, loss: %.2e    " % (batch_idx + 1, train_reader.nbatches, loss_value.numpy()))
            stdout.flush()
            if (batch_idx + 1) % 10 == 0:
                display_training(batch_imgs, flows)
            step_count += 1
        stdout.write('\n')
        print('Epoch computed in ' + str(datetime.now() - epoch_start))

        # Save models:
        print('Saving models')
        flow_net.save_weights(os.path.join(save_dir, 'flow_net_' + str(epoch), 'weights'))
