import tensorflow as tf
from data_reader import DataReader
from model import MyModel
from loss import MyLoss
import Preprocessor
import cv2
import numpy as np
import draw_optical_flow
import datetime
from tensorboard import program
from image_warp import image_warp

# restore_path = r'checkpoints/mymodel'
restore_path = None

kitti_path = r'C:\datasets\KITTI'
sintel_path = r'C:\datasets\MPI-Sintel-complete'
youtube_path = r'C:\datasets\youtube'
# reader = DataReader(kitti_path, sintel_path, youtube_path)
reader = DataReader(kitti_path, None, None)

# This are **roughly** the KITTI camera parameters. But they are not precise, and
# they are not even the same on all sequences!!
fx = 720.0
fy = 720.0
cx = 608.0
cy = 180.0

# I've downsampled the images, so:
fx /= 1.725
fy /= 1.67
cx /= 1.725
cy /= 1.67

model = MyModel(fx, fy, cx, cy)
model.build((None, 224, 720, 6))
model.summary()

if restore_path:
    model.load_weights(restore_path)

train_loss = tf.keras.metrics.Mean(name='train_loss')

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
# optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = 'logs/' + current_time
summary_writer = tf.summary.create_file_writer(log_dir)

# save_path = 'checkpoints\\run_' + current_time
save_path = 'checkpoints/mymodel'

@tf.function
def train_step(images):
    # images: (bs, h, w, 6)
    with tf.GradientTape() as tape:
        predictions = model(images, training=True)
        loss = MyLoss(images, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_loss(loss)

print("TENSORBOARD...")
tb = program.TensorBoard()
tb.configure(argv=[None, '--logdir', log_dir])
url = tb.launch()

batch_size = 8

num_train_steps = 1000000
for step in range(num_train_steps):
    batch = reader.GetNextBatch(batch_size)
    images = Preprocessor.preprocess_image_pairs(batch)

    if (step % 10 == 0):
        # Show first pair:
        im1 = images[0, :, :, :3]
        im2 = images[0, :, :, 3:]
        mean = Preprocessor.mean
        mean = np.squeeze(mean, axis=0)
        img_to_show_1 = im1 + mean
        img_to_show_2 = im2 + mean
        cv2.imshow('image 1', img_to_show_1)
        cv2.imshow('image 2', img_to_show_2)
        # Run network on first pair:
        imgs_to_predict = np.expand_dims(images[0, :, :, :], axis=0)
        flows, motions, depths, egos = model(imgs_to_predict, training=False)
        optical_flow = flows[-1][0, ...]  # (h, w, 2)
        print('optical flow range: ' + str(np.min(optical_flow)) + ' ' + str(np.mean(optical_flow)) + ' ' +
              str(np.max(optical_flow)))
        motion = motions[-1][0, ...]  # (h, w, 3)
        print('motion range: ' + str(np.min(motion)) + ' ' + str(np.mean(motion)) + ' ' +
              str(np.max(motion)))
        depth = depths[-1][0, ...]  # (h, w, 3)
        print('depth range: ' + str(np.min(depth)) + ' ' + str(np.mean(depth)) + ' ' +
              str(np.max(depth)))
        ego = egos[-1][0, ...]  # (h, w, 3)
        print('ego range: ' + str(np.min(ego)) + ' ' + str(np.mean(ego)) + ' ' +
              str(np.max(ego)))
        # Show optical flow:
        optical_flow = optical_flow.numpy()
        arrows_img = draw_optical_flow.draw_all_arrows(im1 + mean, im2 + mean, optical_flow)
        cv2.imshow('Optical flow', arrows_img)
        # itensity_img = draw_optical_flow.draw_optical_flow_intesity(optical_flow)
        # cv2.imshow('Intensity', itensity_img)
        color_img = draw_optical_flow.draw_optical_flow_color(optical_flow)
        cv2.imshow('Flow', color_img)
        # Show warped image:
        im1_ext = np.expand_dims(im1, axis=0)
        optical_flow_ext = np.expand_dims(optical_flow, axis=0)
        im1_warped = image_warp(im1_ext, optical_flow_ext)
        im1_warped += mean
        im1_warped = np.squeeze(im1_warped, axis=0)
        cv2.imshow("im1_warped", im1_warped)
        cv2.waitKey(1)

    train_step(images)

    with summary_writer.as_default():
        tf.summary.scalar('loss', train_loss.result(), step=step)

    print('step ' + str(step) + ' train loss: ' + str(train_loss.result().numpy()))

    if (step % 1000 == 0 and step > 0):
        model.save_weights(save_path)




