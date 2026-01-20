import PIL.Image as Image
import numpy as np
from matplotlib import pyplot as plt

from msnerf.tools.miscs import get_cali_matrix

if __name__ == '__main__':
    img = r"D:\files\PHD\myNeRF\data\83_0\images\0000.png"
    ms_img = np.empty((1085, 2045, 25), dtype=np.uint8)
    ms_idx = np.arange(25).reshape((5, 5))
    ms_idx = np.tile(ms_idx, (1085 // 5, 2048 // 5))[..., None]
    for band in range(25):
        ms_img[..., band] = \
            np.array(Image.open(fr"D:\files\PHD\myNeRF\msnerf\outputs\83_0\msnerf\render\0000_{band}.png"))[
                :1085, :2045]
    ms_img = np.take_along_axis(ms_img, ms_idx, axis=-1)
    ms_img = ms_img.reshape(ms_img.shape[0] // 5, 5, ms_img.shape[1] // 5, 5)
    ms_img = ms_img.transpose((0, 2, 1, 3))
    ms_img = ms_img.reshape(ms_img.shape[0], ms_img.shape[1], 25)

    img = Image.open(img)
    img = np.array(img)[:1085, :2045]
    img = img.reshape(img.shape[0] // 5, 5, img.shape[1] // 5, 5)
    img = img.transpose((0, 2, 1, 3))
    img = img.reshape(img.shape[0], img.shape[1], 25)

    cali = np.eye(25, dtype=np.float32)
    ms_img = ms_img @ cali.T
    img = img @ cali.T
    dot = np.sum(ms_img * img, axis=-1)
    norm_ms = np.linalg.norm(ms_img, axis=-1)
    norm_img = np.linalg.norm(img, axis=-1)
    cos = dot / (norm_ms * norm_img)

    plt.imshow(np.degrees(np.arccos(cos)))
    plt.colorbar()
    plt.show()