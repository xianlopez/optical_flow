import numpy as np
import cv2


def draw_arrow(image, x, y, optical_flow):
    height, width, _ = image.shape
    start_point = np.array([x, y])
    end_point = start_point + optical_flow
    end_point = np.round(end_point).astype(np.int32)
    end_point = np.clip(end_point, 0, [width - 1, height - 1])
    assert end_point[0] < width
    assert end_point[1] < height
    # cv2.arrowedLine(image, tuple(start_point), tuple(end_point), (1.0, 0.0, 0.0))
    cv2.arrowedLine(image, tuple(start_point), tuple(end_point), (1.0, 0.0, 0.0), thickness=2)


def draw_all_arrows(img1, img2, optical_flow):
    assert img1.shape == img2.shape
    height, width, _ = img1.shape
    assert optical_flow.shape[0] == height
    assert optical_flow.shape[1] == width
    assert optical_flow.shape[2] == 2
    assert img1.max() <= 1
    assert img2.max() <= 1
    blended_image = (img1 + img2) * 0.5
    narrows_per_row = 15
    narrows_per_col = 8
    for y in np.arange(0, height, height // narrows_per_col):
        assert y < height
        for x in np.arange(0, width, width // narrows_per_row):
            assert x < width
            draw_arrow(blended_image, x, y, optical_flow[y, x, :])
    # blended_image = cv2.resize(blended_image, (0, 0), fx=2, fy=2)
    return blended_image


def draw_optical_flow_intensity(optical_flow):
    max_length = 20.0
    intensity = np.sqrt(np.square(optical_flow[:, :, 0]) + np.square(optical_flow[:, :, 0]))
    intensity /= max_length
    intensity = np.minimum(intensity, 1.0)
    return intensity


def draw_optical_flow_color(optical_flow):
    height, width, _ = optical_flow.shape
    hsv = np.zeros((height, width, 3), np.uint8)
    hsv[:, :, 1] = 255
    mag, ang = cv2.cartToPolar(optical_flow[..., 0], optical_flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    color_flow = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return color_flow


def display_training(batch_imgs, flows):
    # Display only the first element of the batch:
    img1 = batch_imgs[0, :, :, :3]
    img2 = batch_imgs[1, :, :, 3:]
    flow = flows[0][0, :, :, :].numpy()  # Flow at the maximum scale
    img1_down = cv2.resize(img1, (flow.shape[1], flow.shape[0]))
    img2_down = cv2.resize(img2, (flow.shape[1], flow.shape[0]))
    blended_image = draw_all_arrows(img1_down, img2_down, flow)
    img_to_show = cv2.resize(blended_image, (flow.shape[1] * 4, flow.shape[0] * 4))
    cv2.imshow('flow', img_to_show)
    cv2.waitKey(10)

