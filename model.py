import tensorflow as tf
from resnet.models import resnet_layer_simple
from tensorflow.keras import layers, Sequential, Model, Input, utils
from tensorflow.keras.regularizers import l2
from transformations import bilinear_interpolation

l2_reg = 1e-4


def reset18_encoder(height, width, name):
    input_tensor = Input(shape=(height, width, 3))
    x = layers.Conv2D(64, 7, strides=2, padding='same', name='conv1',
                      kernel_regularizer=l2(l2=l2_reg), bias_regularizer=l2(l2=l2_reg))(input_tensor)
    x = layers.BatchNormalization(name='layer1_bn')(x)
    x = layers.ReLU(name='layer1_relu')(x)
    outputs = [x]  # (batch_size, h/2, w/2, 64)
    x = layers.MaxPool2D(name='layer2_pool')(x)
    x = resnet_layer_simple(x, 2, False, 2)
    outputs.append(x)  # (batch_size, h/4, w/4, 64)
    x = resnet_layer_simple(x, 2, True, 3)
    outputs.append(x)  # (batch_size, h/8, w/8, 128)
    x = resnet_layer_simple(x, 2, True, 4)
    outputs.append(x)  # (batch_size, h/16, w/16, 256)
    # x = resnet_layer_simple(x, 2, True, 5)
    # outputs.append(x)  # (batch_size, h/32, w/32, 512)
    return Model(inputs=input_tensor, outputs=outputs, name=name)


def create_cost_volume(features1, features2, max_disp):
    _, height, width, _ = features1.shape
    assert features1.shape[1] == features2.shape[1]
    assert features1.shape[2] == features2.shape[2]
    assert features1.shape[3] == features2.shape[3]
    features2 = tf.pad(features2, [[0, 0], [max_disp, max_disp], [max_disp, max_disp], [0, 0]])

    correlations = []
    for i in range(2 * max_disp + 1):
        for j in range(2 * max_disp + 1):
            correlations.append(tf.reduce_sum(features1 * features2[:, i:(i+height), j:(j+width), :], axis=-1))
    cost_volume = tf.stack(correlations, axis=-1)  # (batch_size, height, width, (2*max_disp+1)^2)

    return cost_volume


# def flow_from_cost_volume(cost_volume, max_disp):
#     # TODO: argmax or CNN to get displacement? or softmax?
#     best_flat_indices = tf.argmax(cost_volume, axis=-1)  # (batch_size, height, width)
#     indices_i = tf.math.floordiv(best_flat_indices, 2 * max_disp + 1)
#     indices_j = tf.math.floormod(best_flat_indices, 2 * max_disp + 1)
#     indices_ij = tf.stack([indices_i, indices_j], axis=-1)  # (batch_size, height, width, 2)
#     flow_1_to_2 = indices_ij - max_disp
#     return flow_1_to_2


def warp_features(flow_1_to_2, features2):
    # flow_1_to_2: (batch_size, height, width, 2)
    # features2: (batch_size, height, width, nchannels)
    # TODO: Some things here could be done just once
    batch_size = tf.shape(features2)[0]
    _, height, width, _ = features2.shape
    # assert flow_1_to_2.shape[0] == batch_size
    assert flow_1_to_2.shape[1] == height
    assert flow_1_to_2.shape[2] == width
    assert flow_1_to_2.shape[3] == 2
    indices_i, indices_j = tf.meshgrid(tf.range(height), tf.range(width), indexing='ij')  # (height, width)
    indices_ij = tf.stack([indices_i, indices_j], axis=-1)  # (batch_size, height, width, 2)
    aux1 = tf.expand_dims(indices_ij, axis=0)
    aux2 = tf.tile(aux1, [batch_size, 1, 1, 1])
    indices_ij = tf.cast(aux2, tf.float32)  # (bs, h, w, 2)
    # indices_ij = tf.cast(tf.tile(tf.expand_dims(indices_ij, axis=0), [batch_size, 1, 1, 1]), tf.float32)  # (bs, h, w, 2)
    indices_ij = indices_ij + flow_1_to_2
    features_warped = bilinear_interpolation(features2, indices_ij)
    return features_warped  # (batch_size, height, width, nchannels)


def upscale_flow(flow):
    return tf.image.resize(flow, [flow.shape[1] * 2, flow.shape[2] * 2]) * 2.0


def flow_module(features1, features2, previous_flow, max_disp):
    # features1: (bs, h, w, nchannels)
    # features2: (bs, h, w, nchannels)
    assert features1.shape[1] == features2.shape[1]
    assert features1.shape[2] == features2.shape[2]
    assert features1.shape[3] == features2.shape[3]

    if previous_flow is not None:
        assert previous_flow.shape[0] == features1.shape[0]
        assert previous_flow.shape[1] * 2 == features1.shape[1]
        assert previous_flow.shape[2] * 2 == features1.shape[2]
        assert previous_flow.shape[3] == 2
        flow_up = upscale_flow(previous_flow)
        features2_warped = warp_features(flow_up, features2)
    else:
        features2_warped = features2

    cost_volume = create_cost_volume(features1, features2_warped, max_disp)  # (bs, h, w, (2*max_disp+1)^2)

    if previous_flow is not None:
        x = tf.concat([features1, cost_volume, flow_up], axis=-1)
    else:
        x = tf.concat([features1, cost_volume], axis=-1)

    x = layers.Conv2D(128, 3, padding='same', activation=None)(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2D(128, 3, padding='same', activation=None)(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2D(96, 3, padding='same', activation=None)(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2D(64, 3, padding='same', activation=None)(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2D(32, 3, padding='same', activation=None)(x)
    x = layers.LeakyReLU()(x)
    flow = layers.Conv2D(2, 3, padding='same', activation=None)(x)

    return flow


def build_flow_net(height, width, pretrained_weights_path, max_disp=3):
    input_tensor = Input(shape=(height, width, 6))

    image1 = input_tensor[:, :, :, :3]
    image2 = input_tensor[:, :, :, 3:]

    encoder = reset18_encoder(height, width, 'ResNet18')

    if pretrained_weights_path is not None:
        read_result = encoder.load_weights(pretrained_weights_path)
        read_result.assert_existing_objects_matched()

    encoder1_outputs = encoder(image1)
    encoder2_outputs = encoder(image2)

    flow0 = flow_module(encoder1_outputs[-1], encoder2_outputs[-1], None, max_disp)
    flow1 = flow_module(encoder1_outputs[-2], encoder2_outputs[-2], flow0, max_disp)
    flow2 = flow_module(encoder1_outputs[-3], encoder2_outputs[-3], flow1, max_disp)
    flow3 = flow_module(encoder1_outputs[-4], encoder2_outputs[-4], flow2, max_disp)
    # flow4 = flow_module(encoder1_outputs[-5], encoder2_outputs[-5], flow3, max_disp)

    outputs = [flow3, flow2, flow1, flow0]
    # outputs = [flow4, flow3, flow2, flow1, flow0]

    return Model(inputs=input_tensor, outputs=outputs, name="flow_net")


