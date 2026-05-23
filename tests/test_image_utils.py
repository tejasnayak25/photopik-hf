import numpy as np
from utils.image import resize_image, crop_bbox


def test_resize_image_noop():
    img = np.zeros((100, 200, 3), dtype=np.uint8)
    out = resize_image(img, max_width=300)
    assert out.shape == img.shape


def test_resize_image_downscale():
    img = np.zeros((400, 1600, 3), dtype=np.uint8)
    out = resize_image(img, max_width=800)
    assert out.shape[1] == 800


def test_crop_bbox():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    bbox = [10, 10, 50, 60]
    crop = crop_bbox(img, bbox)
    assert crop.shape[0] == 50
    assert crop.shape[1] == 40
