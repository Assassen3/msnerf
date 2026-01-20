import os

import cv2
import tqdm

from msnerf.tools.miscs import get_cali_matrix

src_dir = r"D:\files\PHD\myNeRF\data\87_0\images"
dst_dir = r"D:\files\PHD\myNeRF\msnerf\outputs\87_0\msnerf\spec_correction"
cali = get_cali_matrix().T
for img_name in tqdm.tqdm(os.listdir(src_dir)):
    img_path = os.path.join(src_dir, img_name)
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    img = img[:1085, :2045].reshape((217, 5, 409, 5)).transpose(0, 2, 1, 3).reshape((217, 409, 25))
    img = img @ cali
    for band in range(25):
        cv2.imwrite(os.path.join(dst_dir, f"{img_name}_{band:02d}.png"), img[:, :, band])
