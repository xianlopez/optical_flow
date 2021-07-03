import numpy as np
import random
import cv2

output_height = 224
output_width = 720

r_mean = 0.485
g_mean = 0.456
b_mean = 0.406

mean = np.zeros((1, 1, 1, 3), np.float32)
mean[0, 0, 0, 0] = b_mean
mean[0, 0, 0, 1] = g_mean
mean[0, 0, 0, 2] = r_mean

def preprocess_image_pairs(batch):
    batch_size = len(batch)
    output = np.zeros((batch_size, output_height, output_width, 6), np.float32)
    for i in range(batch_size):
        img1 = batch[i][0]
        img2 = batch[i][1]
        assert img1.shape == img2.shape
        img1 = cv2.resize(img1, (output_width, output_height))
        img2 = cv2.resize(img2, (output_width, output_height))
        img1 = img1 / 255.0 - mean
        img2 = img2 / 255.0 - mean
        output[i, :, :, :3] = img1
        output[i, :, :, 3:] = img2
    return output

