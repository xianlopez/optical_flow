import numpy as np
import cv2


def draw_arrow(image, i, j, optical_flow):
    start_point = np.array([i, j])
    end_point = start_point + optical_flow
    end_point = np.round(end_point).astype(np.int32)
    end_point = np.clip(end_point, 0, [image.shape[0] - 1, image.shape[1] - 1])
    assert end_point[0] < image.shape[0]
    assert end_point[1] < image.shape[1]
    # cv2.arrowedLine(image, tuple(start_point), tuple(end_point), (1.0, 0.0, 0.0))
    cv2.arrowedLine(image, (start_point[1], start_point[0]), (end_point[1], end_point[0]), (1.0, 0.0, 0.0))


def draw_optical_flow(img1, img2, optical_flow):
    assert img1.shape == img2.shape
    height, width, _ = img1.shape
    assert optical_flow.shape[0] == height
    assert optical_flow.shape[1] == width
    assert optical_flow.shape[2] == 2
    assert img1.max() <= 1
    assert img2.max() <= 1
    blended_image = (img1 + img2) * 0.5
    narrows_per_row = 30
    narrows_per_col = 15
    for i in np.arange(0, height, height // narrows_per_col):
        assert i < height
        for j in np.arange(0, width, width // narrows_per_row):
            assert j < width
            draw_arrow(blended_image, i, j, optical_flow[i, j, :])
    blended_image = cv2.resize(blended_image, (0, 0), fx=2, fy=2)
    return blended_image


