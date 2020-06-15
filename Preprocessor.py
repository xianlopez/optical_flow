import numpy as np
import random

output_height = 224
output_width = 224

r_mean = 0.485
g_mean = 0.456
b_mean = 0.406

mean = np.zeros((1, 1, 1, 3), np.float32)
mean[0, 0, 0, 0] = b_mean
mean[0, 0, 0, 1] = g_mean
mean[0, 0, 0, 2] = r_mean

def random_crop_coords(height, width):
    assert height >= output_height
    assert width >= output_width
    max_start_i = height - output_height
    max_start_j = width - output_width
    start_i = random.randint(0, max_start_i)
    start_j = random.randint(0, max_start_j)
    return start_i, start_j

def take_random_crop(full_image, start_i, start_j):
    assert full_image.shape[0] >= start_i+output_height
    assert full_image.shape[1] >= start_j+output_width
    crop = full_image[start_i:(start_i+output_height), start_j:(start_j+output_width)]
    return crop

def preprocess_image_pairs(batch):
    batch_size = len(batch)
    output = np.zeros((batch_size, output_height, output_width, 6), np.float32)
    for i in range(batch_size):
        img1 = batch[i][0]
        img2 = batch[i][1]
        assert img1.shape == img2.shape
        start_i, start_j = random_crop_coords(img1.shape[0], img1.shape[1])
        crop1 = take_random_crop(img1, start_i, start_j)
        crop2 = take_random_crop(img2, start_i, start_j)
        crop1 = crop1 / 255.0 - mean
        crop2 = crop2 / 255.0 - mean
        output[i, :, :, :3] = crop1
        output[i, :, :, 3:] = crop2
    return output

