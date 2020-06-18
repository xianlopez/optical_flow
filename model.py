import tensorflow as tf

from tensorflow.keras import Model, layers

from image_warp import image_warp_tf


class MyModel(Model):
    def __init__(self):
        super(MyModel, self).__init__()

        self.conv1_1 = layers.Conv2D(64, 5, activation='relu', padding='same')
        self.conv1_2 = layers.Conv2D(64, 3, activation='relu', padding='same')
        self.maxpool1 = layers.MaxPool2D()
        self.conv2_1 = layers.Conv2D(128, 5, activation='relu', padding='same')
        self.conv2_2 = layers.Conv2D(128, 3, activation='relu', padding='same')
        self.maxpool2 = layers.MaxPool2D()
        self.conv3_1 = layers.Conv2D(192, 3, activation='relu', padding='same')
        self.conv3_2 = layers.Conv2D(192, 3, activation='relu', padding='same')
        self.maxpool3 = layers.MaxPool2D()
        self.conv4_1 = layers.Conv2D(256, 3, activation='relu', padding='same')
        self.conv4_2 = layers.Conv2D(256, 3, activation='relu', padding='same')
        self.maxpool4 = layers.MaxPool2D()
        self.conv5_1 = layers.Conv2D(512, 3, activation='relu', padding='same')
        self.conv5_2 = layers.Conv2D(512, 3, activation='relu', padding='same')

        self.flow5 = layers.Conv2D(2, 5, activation=None, padding='same')
        self.upsampling5_feat = layers.UpSampling2D(interpolation='bilinear')
        self.upsampling5_flow = layers.UpSampling2D(interpolation='bilinear')

        self.conv6_1 = layers.Conv2D(512, 1, activation='relu', padding='same')
        self.conv6_2 = layers.Conv2D(256, 3, activation='relu', padding='same')
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

        self.conv9_1 = layers.Conv2D(64, 1, activation='relu', padding='same')
        self.conv9_2 = layers.Conv2D(64, 3, activation='relu', padding='same')
        self.flow9 = layers.Conv2D(2, 5, activation=None, padding='same')

    def call(self, x):
        x = self.conv1_1(x)
        conv1_out = self.conv1_2(x)
        print("conv1_out = " + str(conv1_out.shape))
        x = self.maxpool1(conv1_out)
        x = self.conv2_1(x)
        conv2_out = self.conv2_2(x)
        print("conv2_out = " + str(conv2_out.shape))
        x = self.maxpool2(conv2_out)
        x = self.conv3_1(x)
        conv3_out = self.conv3_2(x)
        print("conv3_out = " + str(conv3_out.shape))
        x = self.maxpool3(conv3_out)
        x = self.conv4_1(x)
        conv4_out = self.conv4_2(x)
        print("conv4_out = " + str(conv4_out.shape))
        x = self.maxpool4(conv4_out)
        x = self.conv5_1(x)
        x = self.conv5_2(x)
        print("conv5_out = " + str(x.shape))

        flow5 = self.flow5(x)
        flow5_up = self.upsampling5_flow(flow5)
        x = self.upsampling5_feat(x)
        x = tf.concat([x, conv4_out], axis=-1)
        x = image_warp_tf(x, flow5_up)
        x = tf.concat([x, flow5_up], axis=-1)

        x = self.conv6_1(x)
        x = self.conv6_2(x)
        print("conv6_out = " + str(x.shape))
        flow6 = self.flow6(x)
        flow6_up = self.upsampling6_flow(flow6)
        x = self.upsampling6_feat(x)
        x = tf.concat([x, conv3_out], axis=-1)
        x = image_warp_tf(x, flow6_up)
        x = tf.concat([x, flow6_up], axis=-1)

        x = self.conv7_1(x)
        x = self.conv7_2(x)
        print("conv7_out = " + str(x.shape))
        flow7 = self.flow7(x)
        flow7_up = self.upsampling7_flow(flow7)
        x = self.upsampling7_feat(x)
        x = tf.concat([x, conv2_out], axis=-1)
        x = image_warp_tf(x, flow7_up)
        x = tf.concat([x, flow7_up], axis=-1)

        x = self.conv8_1(x)
        x = self.conv8_2(x)
        print("conv8_out = " + str(x.shape))
        flow8 = self.flow8(x)
        flow8_up = self.upsampling8_flow(flow8)
        x = self.upsampling8_feat(x)
        x = tf.concat([x, conv1_out], axis=-1)
        x = image_warp_tf(x, flow8_up)
        x = tf.concat([x, flow8_up], axis=-1)

        x = self.conv9_1(x)
        x = self.conv9_2(x)
        print("conv9_out = " + str(x.shape))
        flow9 = self.flow9(x)
        print("flow9 = " + str(flow9.shape))

        return [flow5, flow6, flow7, flow8, flow9]









