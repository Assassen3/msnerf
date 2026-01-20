from os.path import join as osj

import cv2
import matplotlib.pyplot as plt
import numpy as np

base_folder = r"D:\files\PHD\myNeRF\paper\Fig\synthetic views"
imgs = ['gt_single.png', '3DGS.png', 'nerf.jpg', 'ours.png']
ul_points = [
    (50, 180),
    (60, 230),
    (120, 120),
    (90, 150)
]
w, h = 90, 50
for i in range(4):
    p = ul_points[i]
    for img_name in imgs:
        img = cv2.imread(osj(base_folder, str(i), img_name))
        roi = [p[0], p[1], p[0] + h, p[1] + w]
        resize = False
        if img.shape[0] == 1088 or img.shape[0] == 1085:
            roi = [_ * 5 for _ in roi]
            resize = True
        img = plt.get_cmap('cividis')(img[..., 0] / 255.0)
        img = (img * 255.0).astype(np.uint8)
        roi_img = img[roi[0]:roi[2], roi[1]:roi[3], :]
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGRA)
        assert (img[:, :, -1] == 255).all()
        # 加一个白边框
        roi_img = cv2.cvtColor(roi_img, cv2.COLOR_RGB2BGRA)
        cv2.rectangle(img, (roi[1], roi[0]), (roi[3], roi[2]), (255, 255, 255), 5 if resize else 1)
        cv2.imwrite(osj(base_folder, str(i), f'c_{img_name}'), img)
        cv2.imwrite(osj(base_folder, str(i), f'cc_{img_name}'), roi_img)

cbar = np.arange(256) / 255.0
cbar = plt.get_cmap('cividis')(cbar) * 255
cbar = cbar.astype(np.uint8)
cbar = cbar[np.newaxis, ...]
cbar = cv2.cvtColor(cbar, cv2.COLOR_RGB2BGRA)
cv2.imwrite(osj(base_folder, 'c.png'), cbar)
