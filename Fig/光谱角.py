import PIL.Image as Image
import numpy as np
from matplotlib import pyplot as plt

from msnerf.tools.miscs import get_cali_matrix

datasets = [61, 83, 87, 90]
img_nums = [11, 12, 176, 355]

for i in range(len(datasets)):
    dataset = datasets[i]
    img_num = img_nums[i]
    img = rf"D:\files\PHD\myNeRF\data\{dataset}_0\images\{img_num:04d}.png"
    ms_idx = np.arange(25).reshape((5, 5))
    ms_idx = np.tile(ms_idx, (1085 // 5, 2048 // 5))[..., None]

    ms_img = np.empty((1085, 2045, 25), dtype=np.uint8)
    for band in range(25):
        ms_img[..., band] = \
            np.array(
                Image.open(fr"D:\files\PHD\myNeRF\msnerf\outputs\{dataset}_0\msnerf\render\{img_num:04d}_{band}.png"))[
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

    cali = get_cali_matrix().T
    cali = np.eye(25)
    ms_img = ms_img @ cali
    img = img @ cali
    dot = np.sum(ms_img * img, axis=-1)
    norm_ms = np.linalg.norm(ms_img, axis=-1)
    norm_img = np.linalg.norm(img, axis=-1)
    cos = dot / (norm_ms * norm_img)
    cos = np.degrees(np.arccos(cos))
    normalized_cos = np.clip(cos, 0, 8) / 8.0
    cmap = plt.get_cmap('cividis')
    normalized_cos = cmap(normalized_cos)
    Image.fromarray((normalized_cos *255).astype(np.uint8)).save(rf'D:\files\PHD\myNeRF\paper\Fig\sam\{dataset}_{img_num:04d}.png')

