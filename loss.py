import tensorflow as tf

from image_warp import image_warp


# def MyLoss(y_true, y_pred):
#     # y_true contains the input images, stacked over the channels dim: (batch_size, height, width, 6)
#     # y_pred contains the optical flow and the occlusion score: (batch_size, height, width, 3)
#     im1 = y_true[:, :, :, :3]  # (bs, h, w, 3)
#     im2 = y_true[:, :, :, 3:]  # (bs, h, w, 3)
#     optical_flow = y_pred[:, :, :, :2]  # (bs, h, w, 2)
#     occlusion = y_pred[:, :, :, 2]  # (bs, h, w)
#
#     # Photometric loss:
#     im1_warped = image_warp(im1, optical_flow)  # (bs, h, w, 3)
#     pixel_diff = im1_warped - im2  # (bs, h, w, 3)
#     pixel_diff_sq_norm = tf.reduce_sum(tf.square(pixel_diff), axis=-1)  # (bs, h, w)
#     pixel_error = tf.sqrt(pixel_diff_sq_norm + 1e-4)  # (bs, h, w)
#     photo_loss = tf.reduce_sum(pixel_error * (1 - occlusion))
#     # tf.print('pixel_error mean: ', tf.reduce_mean(pixel_error))
#
#     # Occlusion loss:
#     occ_loss = 0.1 * tf.reduce_sum(occlusion)
#
#     # tf.print('photo_loss: ', photo_loss)
#     # tf.print('occ_loss: ', occ_loss)
#     # Full loss:
#     loss = photo_loss + occ_loss
#     # TODO: Should I divide by the batch size?
#     batch_size = im1.shape[0]
#     loss = loss / batch_size
#
#     return loss


# def MyLoss(y_true, y_pred):
#     # y_true contains the input images, stacked over the channels dim: (batch_size, height, width, 6)
#     # y_pred contains the optical flow and the occlusion score: (batch_size, height, width, 3)
#     im1 = y_true[:, :, :, :3]  # (bs, h, w, 3)
#     im2 = y_true[:, :, :, 3:]  # (bs, h, w, 3)
#     optical_flow = y_pred[:, :, :, :2]  # (bs, h, w, 2)
#
#     im1_warped = image_warp(im1, optical_flow)  # (bs, h, w, 3)
#     pixel_diff = im1_warped - im2  # (bs, h, w, 3)
#     pixel_diff_sq_norm = tf.reduce_sum(tf.square(pixel_diff), axis=-1)  # (bs, h, w)
#     pixel_error = tf.sqrt(pixel_diff_sq_norm + 1e-4)  # (bs, h, w)
#     loss = tf.reduce_sum(pixel_error)
#     # TODO: Should I divide by the batch size?
#     batch_size = im1.shape[0]
#     loss = loss / batch_size
#
#     return loss


def MyLoss(y_true, y_pred):
    # y_true contains the input images, stacked over the channels dim: (batch_size, height, width, 6)
    # y_pred contains the optical flow and the occlusion score: (batch_size, height, width, 3)
    im1 = y_true[:, :, :, :3]  # (bs, h, w, 3)
    im2 = y_true[:, :, :, 3:]  # (bs, h, w, 3)
    optical_flow = y_pred[:, :, :, :2]  # (bs, h, w, 2)

    # Photometric loss:
    im1_warped = image_warp(im1, optical_flow)  # (bs, h, w, 3)
    pixel_diff = im1_warped - im2  # (bs, h, w, 3)
    pixel_diff_sq_norm = tf.reduce_sum(tf.square(pixel_diff), axis=-1)  # (bs, h, w)
    pixel_error = tf.sqrt(pixel_diff_sq_norm + 1e-4)  # (bs, h, w)
    photo_loss = tf.reduce_sum(pixel_error)

    # Smoothness loss:
    diff_i = optical_flow[:, :-1, :, :] - optical_flow[:, 1:, :, :]  # (bs, h - 1, w, 2)
    diff_j = optical_flow[:, :, :-1, :] - optical_flow[:, :, 1:, :]  # (bs, h, w - 1, 2)
    smoothness_loss = tf.reduce_sum(tf.sqrt(tf.square(diff_i) + 1e-4)) +\
                      tf.reduce_sum(tf.sqrt(tf.square(diff_j) + 1e-4))

    loss = photo_loss + 0.5 * smoothness_loss
    # TODO: Should I divide by the batch size?
    batch_size = im1.shape[0]
    loss = loss / batch_size

    return loss


