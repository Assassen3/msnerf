import PIL.Image as Image
import cv2
import matplotlib.pyplot as plt
import numpy as np

dataset = 89
img_num = 355
img = rf"D:\files\PHD\myNeRF\data\{dataset}_0\images\{img_num:04d}.png"
img = Image.open(img)
img = np.array(img)[:1085, :2045]
img = img.reshape(img.shape[0] // 5, 5, img.shape[1] // 5, 5)
img = img.transpose((0, 2, 1, 3))
img = img.reshape(img.shape[0], img.shape[1], 25)

ms_img = np.empty((1085, 2045, 25), dtype=np.uint8)
for band in range(25):
    ms_img[..., band] = \
        np.array(
            Image.open(fr"D:\files\PHD\myNeRF\msnerf\outputs\{dataset}_0\msnerf\render\{img_num:04d}_{band}.png"))[
            :1085, :2045]
ms_idx = np.arange(25).reshape((5, 5))
ms_idx = np.tile(ms_idx, (1085 // 5, 2048 // 5))[..., None]
ms_img = np.take_along_axis(ms_img, ms_idx, axis=-1)
ms_img = ms_img.reshape(ms_img.shape[0] // 5, 5, ms_img.shape[1] // 5, 5)
ms_img = ms_img.transpose((0, 2, 1, 3))
ms_img = ms_img.reshape(ms_img.shape[0], ms_img.shape[1], 25)

dot = np.sum(1.0 * ms_img * img, axis=-1)
norm_ms = np.linalg.norm(ms_img, axis=-1)
norm_img = np.linalg.norm(img, axis=-1)
cos = dot / (norm_ms * norm_img)
cos = np.degrees(np.arccos(cos))
vmin = 0
vmax = 8
cos = np.clip(cos, vmin, vmax)
cmap = plt.get_cmap('cividis')
cos = cmap((cos - vmin) / (vmax - vmin))
cos = (cos * 255).astype(np.uint8)
cos = cv2.cvtColor(cos, cv2.COLOR_BGRA2RGB)
cv2.imwrite(fr"D:\files\PHD\myNeRF\paper\Fig\ratio\sam.png", cos)
