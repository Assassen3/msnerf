import os
import re
from os.path import join as osj

import cv2
import matplotlib.pyplot as plt
import numpy as np

base_folder = r"D:\files\PHD\myNeRF\paper\Fig\mask"
for i in range(4):
    for img_name in os.listdir(osj(base_folder, str(i))):
        if re.match("\d{4}.png", img_name):
            img = cv2.imread(osj(base_folder, str(i), img_name))
            img = plt.get_cmap('cividis')(img[..., 0] / 255.0)
            img = (img * 255.0).astype(np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGRA)
            cv2.imwrite(osj(base_folder, str(i), f'c_{img_name}'), img)

