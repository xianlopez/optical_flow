import tensorflow as tf
import cv2

from image_warp import image_warp


def MyLoss(y_true, y_pred):
    # y_true contains the input images, stacked over the channels dim: (batch_size, height, width, 6)
    # y_pred is a list with [flow6, flow7, flow8, flow9, flow10], each of this is (bs, h, w, 2)
    im1 = y_true[:, :, :, :3]  # (bs, h, w, 3)
    im2 = y_true[:, :, :, 3:]  # (bs, h, w, 3)

    batch_size, height, width, _ = im1.shape

    im1_d2 = tf.image.resize(im1, (height // 2, width // 2))
    im2_d2 = tf.image.resize(im2, (height // 2, width // 2))
    im1_d4 = tf.image.resize(im1, (height // 4, width // 4))
    im2_d4 = tf.image.resize(im2, (height // 4, width // 4))
    im1_d8 = tf.image.resize(im1, (height // 8, width // 8))
    im2_d8 = tf.image.resize(im2, (height // 8, width // 8))
    im1_d16 = tf.image.resize(im1, (height // 16, width // 16))
    im2_d16 = tf.image.resize(im2, (height // 16, width // 16))

    flows, depths, motions, egos = y_pred
    flow6, flow7, flow8, flow9, flow10 = flows

    assert flow6.shape == (batch_size, height // 16, width // 16, 2)
    assert flow7.shape == (batch_size, height // 8, width // 8, 2)
    assert flow8.shape == (batch_size, height // 4, width // 4, 2)
    assert flow9.shape == (batch_size, height // 2, width // 2, 2)
    assert flow10.shape == (batch_size, height, width, 2)

    im1_warped = image_warp(im1, flow10)  # (bs, h, w, 3)
    pixel_diff = im1_warped - im2  # (bs, h, w, 3)
    pixel_diff_sq_norm = tf.reduce_sum(tf.square(pixel_diff), axis=-1)  # (bs, h, w)
    pixel_error = tf.sqrt(pixel_diff_sq_norm + 1e-4)  # (bs, h, w)
    photo_loss = tf.reduce_sum(pixel_error)

    im1_d2_warped = image_warp(im1_d2, flow9)  # (bs, h / 2, w / 2, 3)
    pixel_diff = im1_d2_warped - im2_d2  # (bs, h / 2, w / 2, 3)
    pixel_diff_sq_norm = tf.reduce_sum(tf.square(pixel_diff), axis=-1)  # (bs, h / 2, w / 2)
    pixel_error = tf.sqrt(pixel_diff_sq_norm + 1e-4)  # (bs, h / 2, w / 2)
    photo_loss += tf.reduce_sum(pixel_error)

    im1_d4_warped = image_warp(im1_d4, flow8)  # (bs, h / 4, w / 4, 3)
    pixel_diff = im1_d4_warped - im2_d4  # (bs, h / 4, w / 4, 3)
    pixel_diff_sq_norm = tf.reduce_sum(tf.square(pixel_diff), axis=-1)  # (bs, h / 4, w / 4)
    pixel_error = tf.sqrt(pixel_diff_sq_norm + 1e-4)  # (bs, h / 4, w / 4)
    photo_loss += tf.reduce_sum(pixel_error)

    im1_d8_warped = image_warp(im1_d8, flow7)  # (bs, h / 8, w / 8, 3)
    pixel_diff = im1_d8_warped - im2_d8  # (bs, h / 8, w / 8, 3)
    pixel_diff_sq_norm = tf.reduce_sum(tf.square(pixel_diff), axis=-1)  # (bs, h / 8, w / 8)
    pixel_error = tf.sqrt(pixel_diff_sq_norm + 1e-4)  # (bs, h / 8, w / 8)
    photo_loss += tf.reduce_sum(pixel_error)

    im1_d16_warped = image_warp(im1_d16, flow6)  # (bs, h / 16, w / 16, 3)
    pixel_diff = im1_d16_warped - im2_d16  # (bs, h / 16, w / 16, 3)
    pixel_diff_sq_norm = tf.reduce_sum(tf.square(pixel_diff), axis=-1)  # (bs, h / 16, w / 16)
    pixel_error = tf.sqrt(pixel_diff_sq_norm + 1e-4)  # (bs, h / 16, w / 16)
    photo_loss += tf.reduce_sum(pixel_error)

    # Smoothness loss:
    diff_i = flow10[:, :-1, :, :] - flow10[:, 1:, :, :]  # (bs, h - 1, w, 2)
    diff_j = flow10[:, :, :-1, :] - flow10[:, :, 1:, :]  # (bs, h, w - 1, 2)
    smoothness_loss = tf.reduce_sum(tf.sqrt(tf.square(diff_i) + 1e-4)) +\
                      tf.reduce_sum(tf.sqrt(tf.square(diff_j) + 1e-4))

    diff_i = flow9[:, :-1, :, :] - flow9[:, 1:, :, :]  # (bs, h / 2 - 1, w / 2, 2)
    diff_j = flow9[:, :, :-1, :] - flow9[:, :, 1:, :]  # (bs, h / 2, w / 2 - 1, 2)
    smoothness_loss += tf.reduce_sum(tf.sqrt(tf.square(diff_i) + 1e-4)) +\
                       tf.reduce_sum(tf.sqrt(tf.square(diff_j) + 1e-4))

    diff_i = flow8[:, :-1, :, :] - flow8[:, 1:, :, :]  # (bs, h / 4 - 1, w / 4, 2)
    diff_j = flow8[:, :, :-1, :] - flow8[:, :, 1:, :]  # (bs, h / 4, w / 4 - 1, 2)
    smoothness_loss += tf.reduce_sum(tf.sqrt(tf.square(diff_i) + 1e-4)) +\
                       tf.reduce_sum(tf.sqrt(tf.square(diff_j) + 1e-4))

    diff_i = flow7[:, :-1, :, :] - flow7[:, 1:, :, :]  # (bs, h / 8 - 1, w / 8, 2)
    diff_j = flow7[:, :, :-1, :] - flow7[:, :, 1:, :]  # (bs, h / 8, w / 8 - 1, 2)
    smoothness_loss += tf.reduce_sum(tf.sqrt(tf.square(diff_i) + 1e-4)) +\
                       tf.reduce_sum(tf.sqrt(tf.square(diff_j) + 1e-4))

    diff_i = flow6[:, :-1, :, :] - flow6[:, 1:, :, :]  # (bs, h / 16 - 1, w / 16, 2)
    diff_j = flow6[:, :, :-1, :] - flow6[:, :, 1:, :]  # (bs, h / 16, w / 16 - 1, 2)
    smoothness_loss += tf.reduce_sum(tf.sqrt(tf.square(diff_i) + 1e-4)) +\
                       tf.reduce_sum(tf.sqrt(tf.square(diff_j) + 1e-4))

    loss = photo_loss + 0.2 * smoothness_loss
    # TODO: Should I divide by the batch size?
    loss = loss / batch_size

    return loss


