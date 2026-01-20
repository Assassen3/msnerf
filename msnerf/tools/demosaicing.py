from pathlib import Path

import cv2
from tqdm import tqdm

if __name__ == "__main__":
    # --- 配置区域 ---
    INPUT_FOLDER = Path(r"D:\files\PHD\myNeRF\data\61_0\images")
    OUTPUT_FOLDER = Path(r"D:\files\PHD\myNeRF\data\61_0\demosaic_images")
    MOSAIC_SIZE = 5
    FILTER_METHOD = 'box'  # 推荐使用 'box' (均值) 针对这种规律性马赛克

    for img in tqdm(INPUT_FOLDER.iterdir()):
        if img.suffix != '.png':
            continue
        img_np = cv2.imread(str(img), cv2.IMREAD_UNCHANGED)

        res_gauss = cv2.blur(img_np, (5, 5))

        cv2.imwrite(str(OUTPUT_FOLDER / img.name), res_gauss)
