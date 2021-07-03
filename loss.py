import tensorflow as tf
from model import warp_features


class LossLayer:
    def __call__(self, batch_imgs, flows):
        num_scales = len(flows)
        loss = tf.zeros((), tf.float32)

        for scale_idx in range(num_scales):
            scaled_height = int(batch_imgs.shape[1] / (2.0 ** (scale_idx + 1)))
            scaled_width = int(batch_imgs.shape[2] / (2.0 ** (scale_idx + 1)))
            flow = flows[scale_idx]  # (bs, scaled_height, scaled_width, 2)
            assert flow.shape[1] == scaled_height
            assert flow.shape[2] == scaled_width

            if scaled_height != batch_imgs.shape[1] or scaled_width != batch_imgs.shape[2]:
                resized_imgs = tf.image.resize(batch_imgs, (scaled_height, scaled_width))
                assert resized_imgs.shape[0] == batch_imgs.shape[0]
                assert resized_imgs.shape[1] == scaled_height
                assert resized_imgs.shape[2] == scaled_width
                assert resized_imgs.shape[3] == 6
            else:
                resized_imgs = batch_imgs

            image2_warped = warp_features(flow, resized_imgs[:, :, :, 3:])

            loss += tf.reduce_mean(tf.math.abs(resized_imgs[:, :, :, :3] - image2_warped))

        loss /= float(num_scales)

        return loss


