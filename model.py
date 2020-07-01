import tensorflow as tf
import numpy as np

from tensorflow.keras import Model, layers

from image_warp import image_warp_tf


def print_mean_std(x, name):
    mean = tf.reduce_mean(x)
    centered_sq = tf.square(x - mean)
    std = tf.sqrt(tf.reduce_mean(centered_sq))
    tf.print(name + ' mean = ', mean)
    tf.print(name + ' std = ', std)


def rotation_x(angle):
    # angle: bs
    assert len(angle.shape) == 1
    bs = tf.unstack(tf.shape(angle))[0]
    pos00 = tf.cos(angle)
    pos01 = -tf.sin(angle)
    pos10 = tf.sin(angle)
    pos11 = tf.cos(angle)
    ones = tf.repeat(1.0, bs)
    zeros = tf.repeat(0.0, bs)
    R_col0 = tf.stack([pos00, pos10, zeros], axis=-1)  # bs, 3
    R_col1 = tf.stack([pos01, pos11, zeros], axis=-1)  # bs, 3
    R_col2 = tf.stack([zeros, zeros, ones], axis=-1)  # bs, 3
    R = tf.stack([R_col0, R_col1, R_col2], axis=-1)  # bs, 3, 3
    return R


def rotation_y(angle):
    # angle: bs
    assert len(angle.shape) == 1
    bs = tf.unstack(tf.shape(angle))[0]
    pos00 = tf.cos(angle)
    pos02 = tf.sin(angle)
    pos20 = -tf.sin(angle)
    pos22 = tf.cos(angle)
    ones = tf.repeat(1.0, bs)
    zeros = tf.repeat(0.0, bs)
    R_col0 = tf.stack([pos00, zeros, pos20], axis=-1)  # bs, 3
    R_col1 = tf.stack([zeros, ones, zeros], axis=-1)  # bs, 3
    R_col2 = tf.stack([pos02, zeros, pos22], axis=-1)  # bs, 3
    R = tf.stack([R_col0, R_col1, R_col2], axis=-1)  # bs, 3, 3
    return R


def rotation_z(angle):
    # angle: bs
    assert len(angle.shape) == 1
    bs = tf.unstack(tf.shape(angle))[0]
    pos11 = tf.cos(angle)  # bs
    pos12 = -tf.sin(angle)  # bs
    pos21 = tf.sin(angle)  # bs
    pos22 = tf.cos(angle)  # bs
    ones = tf.repeat(1.0, bs)  # bs
    zeros = tf.repeat(0.0, bs)  # bs
    R_col0 = tf.stack([ones, zeros, zeros], axis=-1)  # bs, 3
    R_col1 = tf.stack([zeros, pos11, pos21], axis=-1)  # bs, 3
    R_col2 = tf.stack([zeros, pos12, pos22], axis=-1)  # bs, 3
    R = tf.stack([R_col0, R_col1, R_col2], axis=-1)  # bs, 3, 3
    return R


def compute_flow(depth, motion, ego, fx, fy):
    # depth: bs, h, w, 1
    # motion: bs, h, w, 3
    # ego: bs, 6
    bs = tf.unstack(tf.shape(depth))[0]
    _, h, w, _ = depth.shape
    cx = w / 2.0
    cy = h / 2.0
    assert len(motion.shape) == 4
    assert len(ego.shape) == 2
    assert motion.shape[1] == h
    assert motion.shape[2] == w
    assert motion.shape[3] == 3
    assert ego.shape[1] == 6
    # TODO: Image coordinates 0-based or 1-based?
    x1_1, x1_2 = tf.meshgrid(np.arange(w), np.arange(h))  # h, w
    x1_1 = tf.cast(tf.tile(tf.expand_dims(x1_1, axis=0), [bs, 1, 1]), tf.float32)  # bs, h, w
    x1_2 = tf.cast(tf.tile(tf.expand_dims(x1_2, axis=0), [bs, 1, 1]), tf.float32)  # bs, h, w
    rotation = tf.matmul(rotation_z(ego[:, 2]),
                         tf.matmul(rotation_y(ego[:, 1]), rotation_x(ego[:, 0])))  # bs, 3, 3
    rotation = tf.tile(tf.reshape(rotation, (bs, 1, 1, 3, 3)), [1, h, w, 1, 1])  # bs, h, w, 3, 3
    X1_1 = depth / fx * (tf.expand_dims(x1_1, axis=-1) - cx)
    X1_2 = depth / fy * (tf.expand_dims(x1_2, axis=-1) - cy)
    X1 = tf.concat([X1_1, X1_2, depth], axis=-1)  # bs, h, w, 3
    translation = tf.tile(tf.reshape(ego[:, 3:], (bs, 1, 1, 3)), [1, h, w, 1])  # bs, h, w, 3
    X1_moved = X1 + motion
    X1_rotated = tf.squeeze(tf.matmul(rotation, tf.expand_dims(X1_moved, axis=-1)), axis=-1)
    X2 = X1_rotated + translation  # bs, h, w, 3
    x2_1 = fx * X2[:, :, :, 0] / X2[:, :, :, 2] + cx  # bs, h, w
    x2_2 = fy * X2[:, :, :, 1] / X2[:, :, :, 2] + cy  # bs, h, w
    flowX = x2_1 - x1_1
    flowY = x2_2 - x1_2
    flow = tf.stack([flowX, flowY], axis=-1)
    return flow


class MyModel(Model):
    def __init__(self, fx, fy, cx, cy):
        super(MyModel, self).__init__()

        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy

        self.conv1_1 = layers.Conv2D(32, 5, activation='relu', padding='same', name='conv1_1')
        self.conv1_2 = layers.Conv2D(32, 3, activation='relu', padding='same', name='conv2_1')
        self.maxpool1 = layers.MaxPool2D()
        self.conv2_1 = layers.Conv2D(64, 5, activation='relu', padding='same', name='conv2_1')
        self.conv2_2 = layers.Conv2D(64, 3, activation='relu', padding='same', name='conv2_2')
        self.maxpool2 = layers.MaxPool2D()
        self.conv3_1 = layers.Conv2D(96, 3, activation='relu', padding='same', name='conv3_1')
        self.conv3_2 = layers.Conv2D(96, 3, activation='relu', padding='same', name='conv3_2')
        self.maxpool3 = layers.MaxPool2D()
        self.conv4_1 = layers.Conv2D(128, 3, activation='relu', padding='same', name='conv4_1')
        self.conv4_2 = layers.Conv2D(128, 3, activation='relu', padding='same', name='conv4_2')
        self.maxpool4 = layers.MaxPool2D()
        self.conv5_1 = layers.Conv2D(160, 3, activation='relu', padding='same', name='conv5_1')
        self.conv5_2 = layers.Conv2D(160, 3, activation='relu', padding='same', name='conv5_2')

        self.conv6_1 = layers.Conv2D(256, 1, activation='relu', padding='same', name='conv6_1')
        self.conv6_2 = layers.Conv2D(160, 3, activation='relu', padding='same', name='conv6_2')
        self.motion_6 = layers.Conv2D(3, 5, activation=None, padding='same', name='motion_6')
        self.depth_6 = layers.Conv2D(1, 5, activation=None, padding='same', name='depth_6')

        self.conv7_1 = layers.Conv2D(160, 1, activation='relu', padding='same', name='conv7_1')
        self.conv7_2 = layers.Conv2D(128, 3, activation='relu', padding='same', name='conv7_2')
        self.motion_7 = layers.Conv2D(3, 5, activation=None, padding='same', name='motion_7')
        self.depth_7 = layers.Conv2D(1, 5, activation=None, padding='same', name='depth_7')

        self.conv8_1 = layers.Conv2D(128, 1, activation='relu', padding='same', name='conv8_1')
        self.conv8_2 = layers.Conv2D(64, 3, activation='relu', padding='same', name='conv8_2')
        self.motion_8 = layers.Conv2D(3, 5, activation=None, padding='same', name='motion_8')
        self.depth_8 = layers.Conv2D(1, 5, activation=None, padding='same', name='depth_8')

        self.conv9_1 = layers.Conv2D(128, 1, activation='relu', padding='same', name='conv9_1')
        self.conv9_2 = layers.Conv2D(64, 3, activation='relu', padding='same', name='conv9_2')
        self.motion_9 = layers.Conv2D(3, 5, activation=None, padding='same', name='motion_9')
        self.depth_9 = layers.Conv2D(1, 5, activation=None, padding='same', name='depth_9')

        self.conv10_1 = layers.Conv2D(128, 1, activation='relu', padding='same', name='conv10_1')
        self.conv10_2 = layers.Conv2D(64, 3, activation='relu', padding='same', name='conv10_2')
        self.motion_10 = layers.Conv2D(3, 5, activation=None, padding='same', name='motion_10')
        self.depth_10 = layers.Conv2D(1, 5, activation=None, padding='same', name='depth_10')

        self.upsampling = layers.UpSampling2D(interpolation='bilinear')
        self.flatten = layers.Flatten()

        self.ego_conv1 = layers.Conv2D(128, 3, activation='relu', padding='same', name='ego_conv1')
        self.ego_maxpool1 = layers.MaxPool2D()
        self.ego_conv2 = layers.Conv2D(128, 3, activation='relu', padding='same', name='ego_conv2')
        self.ego_conv3 = layers.Conv2D(64, 3, strides=(2, 2), activation='relu', padding='same', name='ego_conv3')
        self.ego_dense1 = layers.Dense(64, activation='relu', name='ego_dense1')
        self.ego_dense2 = layers.Dense(6, activation=None, name='ego_dense2')

    def call(self, x):
        im1 = x[:, :, :, :3]
        im2 = x[:, :, :, 3:]

        x1 = self.conv1_1(im1)
        im1_conv1_out = self.conv1_2(x1)
        x1 = self.maxpool1(im1_conv1_out)
        x1 = self.conv2_1(x1)
        im1_conv2_out = self.conv2_2(x1)
        x1 = self.maxpool2(im1_conv2_out)
        x1 = self.conv3_1(x1)
        im1_conv3_out = self.conv3_2(x1)
        x1 = self.maxpool3(im1_conv3_out)
        x1 = self.conv4_1(x1)
        im1_conv4_out = self.conv4_2(x1)
        x1 = self.maxpool4(im1_conv4_out)
        x1 = self.conv5_1(x1)
        x1 = self.conv5_2(x1)

        x2 = self.conv1_1(im2)
        im2_conv1_out = self.conv1_2(x2)
        x2 = self.maxpool1(im2_conv1_out)
        x2 = self.conv2_1(x2)
        im2_conv2_out = self.conv2_2(x2)
        x2 = self.maxpool2(im2_conv2_out)
        x2 = self.conv3_1(x2)
        im2_conv3_out = self.conv3_2(x2)
        x2 = self.maxpool3(im2_conv3_out)
        x2 = self.conv4_1(x2)
        im2_conv4_out = self.conv4_2(x2)
        x2 = self.maxpool4(im2_conv4_out)
        x2 = self.conv5_1(x2)
        x2 = self.conv5_2(x2)

        x = tf.concat([x1, x2], axis=-1)  # 14 x 45
        x = self.conv6_1(x)
        x = self.conv6_2(x)
        ego = self.compute_ego(x)

        motion6 = self.motion_6(x) * 0.01
        depth6 = tf.maximum(self.depth_6(x) + 5.0, 0.1)
        flow6 = compute_flow(depth6, motion6, ego, self.fx / 16.0, self.fy / 16.0)  # 14 x 45
        # print('flow6: ' + str(flow6.shape))
        # print_mean_std(motion6, 'motion6')
        # print_mean_std(depth6, 'depth6')
        # print_mean_std(flow6, 'flow6')
        flow6_up = 2 * self.upsampling(flow6)  # 28 x 90
        im1_conv4_out_warped = image_warp_tf(im1_conv4_out, flow6_up)  # 28 x 90
        x = tf.concat([x, motion6, depth6], axis=-1)
        x = self.upsampling(x)  # 28 x 90
        x = tf.concat([x, im1_conv4_out_warped, im2_conv4_out, flow6_up], axis=-1)  # 28 x 90

        x = self.conv7_1(x)  # 28 x 90
        x = self.conv7_2(x)
        motion7 = self.motion_7(x) * 0.01
        depth7 = tf.maximum(self.depth_7(x) + 5.0, 0.1)
        flow7 = compute_flow(depth7, motion7, ego, self.fx / 8.0, self.fy / 8.0)  # 28 x 90
        # print('flow7: ' + str(flow7.shape))
        # print_mean_std(motion7, 'motion7')
        # print_mean_std(depth7, 'depth7')
        # print_mean_std(flow7, 'flow7')
        flow7_up = 2 * self.upsampling(flow7)  # 56 x 180
        im1_conv3_out_warped = image_warp_tf(im1_conv3_out, flow7_up)  # 56 x 180
        x = tf.concat([x, motion7, depth7], axis=-1)  # 28 x 90
        x = self.upsampling(x)  # 56 x 180
        x = tf.concat([x, im1_conv3_out_warped, im2_conv3_out, flow7_up], axis=-1)  # 56 x 180

        x = self.conv8_1(x)  # 56 x 180
        x = self.conv8_2(x)
        motion8 = self.motion_8(x) * 0.01
        depth8 = tf.maximum(self.depth_8(x) + 5.0, 0.1)
        flow8 = compute_flow(depth8, motion8, ego, self.fx / 4.0, self.fy / 4.0)  # 56 x 180
        # print('flow8: ' + str(flow8.shape))
        # print_mean_std(motion8, 'motion8')
        # print_mean_std(depth8, 'depth8')
        # print_mean_std(flow8, 'flow8')
        flow8_up = 2 * self.upsampling(flow8)  # 112 x 360
        im1_conv2_out_warped = image_warp_tf(im1_conv2_out, flow8_up)  # 112 x 360
        x = tf.concat([x, motion8, depth8], axis=-1)  # 56 x 180
        x = self.upsampling(x)  # 112 x 360
        x = tf.concat([x, im1_conv2_out_warped, im2_conv2_out, flow8_up], axis=-1)  # 112 x 360

        x = self.conv9_1(x)  # 112 x 360
        x = self.conv9_2(x)
        motion9 = self.motion_9(x) * 0.01
        depth9 = tf.maximum(self.depth_9(x) + 5.0, 0.1)
        flow9 = compute_flow(depth9, motion9, ego, self.fx / 2.0, self.fy / 2.0)  # 112 x 360
        # print('flow9: ' + str(flow9.shape))
        # print_mean_std(motion9, 'motion9')
        # print_mean_std(depth9, 'depth9')
        # print_mean_std(flow9, 'flow9')
        flow9_up = 2 * self.upsampling(flow9)  # 224 x 720
        im1_conv1_out_warped = image_warp_tf(im1_conv1_out, flow9_up)  # 224 x 720
        x = tf.concat([x, motion9, depth9], axis=-1)  # 112 x 360
        x = self.upsampling(x)  # 224 x 720
        x = tf.concat([x, im1_conv1_out_warped, im2_conv1_out, flow9_up], axis=-1)  # 224 x 720

        x = self.conv10_1(x)  # 224 x 720
        x = self.conv10_2(x)
        motion10 = self.motion_10(x) * 0.01
        depth10 = tf.maximum(self.depth_10(x) + 5.0, 0.1)
        flow10 = compute_flow(depth10, motion10, ego, self.fx, self.fy)  # 224 x 720
        # print('flow10: ' + str(flow10.shape))
        # print_mean_std(motion10, 'motion10')
        # print_mean_std(depth10, 'depth10')
        # print_mean_std(flow10, 'flow10')

        flows = [flow6, flow7, flow8, flow9, flow10]
        motions = [motion6, motion7, motion8, motion9, motion10]
        depths = [depth6, depth7, depth8, depth9, depth10]

        return [flows, motions, depths, ego]

    def compute_ego(self, x):
        assert x.shape[1] == 14
        assert x.shape[2] == 45
        x = self.ego_conv1(x)
        x = self.ego_maxpool1(x)  # 7, 22
        x = self.ego_conv2(x)
        x = self.ego_conv3(x)  # 4, 11
        x = self.flatten(x)
        x = self.ego_dense1(x)
        x = self.ego_dense2(x)
        x = x * 0.1  # To begin with a transformation closer to identity
        angles = tf.clip_by_value(x[:, :3], -np.pi, np.pi)
        translation = x[:, 3:]
        ego = tf.concat([angles, translation], axis=-1)
        return ego









