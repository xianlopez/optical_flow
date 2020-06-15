import tensorflow as tf
from data_reader import DataReader
from model import MyModel
from loss import MyLoss
import Preprocessor
import cv2
import numpy as np
from draw_optical_flow import draw_optical_flow
import datetime
from tensorboard import program

kitti_path = r'C:\datasets\KITTI'

reader = DataReader(kitti_path)

model = MyModel()

model.build((None, 224, 224, 6))
model.summary()

train_loss = tf.keras.metrics.Mean(name='train_loss')

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = 'logs/' + current_time
summary_writer = tf.summary.create_file_writer(log_dir)

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

batch_size = 16

num_train_steps = 100000
for step in range(num_train_steps):
    batch = reader.GetNextBatch(batch_size)
    images = Preprocessor.preprocess_image_pairs(batch)

    if (step % 10 == 0):
        # Show first pair:
        img_to_show_1 = images[0, :, :, :3]
        img_to_show_2 = images[0, :, :, 3:]
        mean = Preprocessor.mean
        mean = np.squeeze(mean, axis=0)
        img_to_show_1 += mean
        img_to_show_2 += mean
        cv2.imshow('image 1', img_to_show_1)
        cv2.imshow('image 2', img_to_show_2)
        # Run network on first pair:
        imgs_to_predict = np.expand_dims(images[0, :, :, :], axis=0)
        predictions = model(imgs_to_predict, training=False)  # (1, h, w, 3)
        optical_flow = predictions[0, :, :, :2]  # (h, w, 2)
        occlusion = predictions[0, :, :, 2]  # (h, w)
        print('optical flow range: ' + str(np.min(optical_flow)) + ' ' + str(np.mean(optical_flow)) + ' ' +
              str(np.max(optical_flow)))
        # print('occlusion range: ' + str(np.min(occlusion)) + ' ' + str(np.mean(occlusion)) + ' ' +
        #       str(np.max(occlusion)))
        # # Show occlusion mask:
        # occlusion = occlusion.numpy()
        # cv2.imshow('occlusion', occlusion)
        # Show optical flow:
        optical_flow = optical_flow.numpy()
        ofx = optical_flow[:, :, 0]
        ofy = optical_flow[:, :, 1]
        max_displacement = 10.0
        zeros = np.zeros_like(ofx)
        ofx_negative = -np.clip(ofx, -max_displacement, 0.0)
        ofx_positive = np.clip(ofx, 0.0, max_displacement)
        color_x = np.stack([ofx_negative / max_displacement, zeros, ofx_positive / max_displacement], axis=-1)
        cv2.imshow('X displacement', color_x)  # Blue: to the left. Red: to the rigth.
        ofy_negative = -np.clip(ofy, -max_displacement, 0.0)
        ofy_positive = np.clip(ofy, 0.0, max_displacement)
        color_y = np.stack([ofy_negative / max_displacement, zeros, ofy_positive / max_displacement], axis=-1)
        cv2.imshow('Y displacement', color_y)  # Blue: downwards. Red: upwards.
        arrows_img = draw_optical_flow(images[0, :, :, :3], images[0, :, :, 3:], optical_flow)
        cv2.imshow('Optical flow', arrows_img)
        cv2.waitKey(1)

    train_step(images)

    with summary_writer.as_default():
        tf.summary.scalar('loss', train_loss.result(), step=step)

    print('step ' + str(step) + ' train loss: ' + str(train_loss.result().numpy()))




