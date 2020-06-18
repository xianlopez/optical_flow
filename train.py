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
from image_warp import image_warp

restore_path = r'checkpoints/mymodel'
# restore_path = None

kitti_path = r'C:\datasets\KITTI'

reader = DataReader(kitti_path)

model = MyModel()
model.build((None, 224, 224, 6))
model.summary()

if restore_path:
    model.load_weights(restore_path)

train_loss = tf.keras.metrics.Mean(name='train_loss')

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

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

batch_size = 16

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
        predictions = model(imgs_to_predict, training=False)  # (1, h, w, 3)
        optical_flow = predictions[-1][0, ...]  # (h, w, 2)
        print('optical flow range: ' + str(np.min(optical_flow)) + ' ' + str(np.mean(optical_flow)) + ' ' +
              str(np.max(optical_flow)))
        # Show optical flow:
        optical_flow = optical_flow.numpy()
        arrows_img = draw_optical_flow(im1 + mean, im2 + mean, optical_flow)
        cv2.imshow('Optical flow', arrows_img)
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

    # if (step % 10 == 0):
    #     print("weight (0, 0, 0, 0) of layer conv2d_22: " +
    #           str(model.get_layer('conv2d_22').weights[0][0, 0, 0, 0].numpy()))




