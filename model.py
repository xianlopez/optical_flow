import tensorflow as tf

from tensorflow.keras import Model, layers

from image_warp import image_warp_tf


class MyModel(Model):
    def __init__(self):
        super(MyModel, self).__init__()

        self.conv1_1 = layers.Conv2D(32, 5, activation='relu', padding='same')
        self.conv1_2 = layers.Conv2D(32, 3, activation='relu', padding='same')
        self.maxpool1 = layers.MaxPool2D()
        self.conv2_1 = layers.Conv2D(64, 5, activation='relu', padding='same')
        self.conv2_2 = layers.Conv2D(64, 3, activation='relu', padding='same')
        self.maxpool2 = layers.MaxPool2D()
        self.conv3_1 = layers.Conv2D(96, 3, activation='relu', padding='same')
        self.conv3_2 = layers.Conv2D(96, 3, activation='relu', padding='same')
        self.maxpool3 = layers.MaxPool2D()
        self.conv4_1 = layers.Conv2D(128, 3, activation='relu', padding='same')
        self.conv4_2 = layers.Conv2D(128, 3, activation='relu', padding='same')
        self.maxpool4 = layers.MaxPool2D()
        self.conv5_1 = layers.Conv2D(256, 3, activation='relu', padding='same')
        self.conv5_2 = layers.Conv2D(256, 1, activation='relu', padding='same')

        self.conv6 = layers.Conv2D(256, 3, activation='relu', padding='same')
        self.flow6 = layers.Conv2D(2, 5, activation=None, padding='same')
        self.upsampling6_feat = layers.UpSampling2D(interpolation='bilinear')
        self.upsampling6_flow = layers.UpSampling2D(interpolation='bilinear')

        self.conv7_1 = layers.Conv2D(256, 1, activation='relu', padding='same')
        self.conv7_2 = layers.Conv2D(128, 3, activation='relu', padding='same')
        self.flow7 = layers.Conv2D(2, 5, activation=None, padding='same')
        self.upsampling7_feat = layers.UpSampling2D(interpolation='bilinear')
        self.upsampling7_flow = layers.UpSampling2D(interpolation='bilinear')

        self.conv8_1 = layers.Conv2D(128, 1, activation='relu', padding='same')
        self.conv8_2 = layers.Conv2D(64, 3, activation='relu', padding='same')
        self.flow8 = layers.Conv2D(2, 5, activation=None, padding='same')
        self.upsampling8_feat = layers.UpSampling2D(interpolation='bilinear')
        self.upsampling8_flow = layers.UpSampling2D(interpolation='bilinear')

        self.conv9_1 = layers.Conv2D(128, 1, activation='relu', padding='same')
        self.conv9_2 = layers.Conv2D(64, 3, activation='relu', padding='same')
        self.flow9 = layers.Conv2D(2, 5, activation=None, padding='same')
        self.upsampling9_feat = layers.UpSampling2D(interpolation='bilinear')
        self.upsampling9_flow = layers.UpSampling2D(interpolation='bilinear')

        self.conv10_1 = layers.Conv2D(64, 1, activation='relu', padding='same')
        self.conv10_2 = layers.Conv2D(64, 3, activation='relu', padding='same')
        self.flow10 = layers.Conv2D(2, 5, activation=None, padding='same')

    def call(self, x):

        im1 = x[:, :, :, :3]
        im2 = x[:, :, :, 3:]
        print("im1: " + str(im1.shape))
        print("im2: " + str(im2.shape))

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
        print("im1 encoded: " + str(x1.shape))

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
        print("im2 encoded: " + str(x2.shape))

        x = tf.concat([x1, x2], axis=-1)
        x = self.conv6(x)
        flow6 = self.flow6(x)
        print("flow6: " + str(flow6.shape))
        flow6_up = 2 * self.upsampling6_flow(flow6)
        x = self.upsampling6_feat(x)
        im1_conv4_out_warped = image_warp_tf(im1_conv4_out, flow6_up)
        x = tf.concat([x, im1_conv4_out_warped, im2_conv4_out, flow6_up], axis=-1)

        x = self.conv7_1(x)
        x = self.conv7_2(x)
        flow7 = self.flow7(x)
        print("flow7: " + str(flow7.shape))
        flow7_up = 2 * self.upsampling7_flow(flow7)
        x = self.upsampling7_feat(x)
        im1_conv3_out_warped = image_warp_tf(im1_conv3_out, flow7_up)
        x = tf.concat([x, im1_conv3_out_warped, im2_conv3_out, flow7_up], axis=-1)

        x = self.conv8_1(x)
        x = self.conv8_2(x)
        flow8 = self.flow8(x)
        print("flow8: " + str(flow8.shape))
        flow8_up = 2 * self.upsampling8_flow(flow8)
        x = self.upsampling8_feat(x)
        im1_conv2_out_warped = image_warp_tf(im1_conv2_out, flow8_up)
        x = tf.concat([x, im1_conv2_out_warped, im2_conv2_out, flow8_up], axis=-1)

        x = self.conv9_1(x)
        x = self.conv9_2(x)
        flow9 = self.flow9(x)
        print("flow9: " + str(flow9.shape))
        flow9_up = 2 * self.upsampling9_flow(flow9)
        x = self.upsampling9_feat(x)
        im1_conv1_out_warped = image_warp_tf(im1_conv1_out, flow9_up)
        x = tf.concat([x, im1_conv1_out_warped, im2_conv1_out, flow9_up], axis=-1)

        x = self.conv10_1(x)
        x = self.conv10_2(x)
        flow10 = self.flow10(x)
        print("flow10: " + str(flow10.shape))

        return [flow6, flow7, flow8, flow9, flow10]









