import json

import cv2
import numpy as np
import torch

reference_gt_reflectance = np.array([
    0.934095, 0.93359506, 0.92347396, 0.91781205, 0.91406208, 0.90738468, 0.90377151, 0.90011352, 0.89591007,
    0.89373012, 0.89181359, 0.88820379, 0.88641948, 0.88421483, 0.88149024, 0.87939531, 0.8786942, 0.87611266,
    0.87441173, 0.87370804, 0.87116529, 0.87044771, 0.87070595, 0.86735694, 0.86614593
], dtype=np.float32).reshape(25, 1)

wavelengths = np.array([667, 670, 689, 702, 713,
                        730, 741, 754, 769, 781,
                        790, 805, 815, 828, 843,
                        853, 864, 878, 890, 898,
                        911, 919, 929, 939, 949], dtype=np.int64)


def get_cali_matrix():
    with open(r"D:\files\PHD\myNeRF\data\metadata\calibration.json", 'r') as f:
        cali_metadata = json.load(f)['calibration']['matrix']
    return np.array(cali_metadata, dtype=np.float32).reshape(25, 25)


def r2_score(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        y_pred = y_pred.float()
        y_true = y_true.float()

        y_true_mean = torch.mean(y_true, dim=0, keepdim=True)

        ss_res = torch.sum((y_true - y_pred) ** 2)
        ss_tot = torch.sum((y_true - y_true_mean) ** 2)

        # 避免除以 0
        return 1 - ss_res / ss_tot


def cmap(x: np.ndarray, type=np.uint8) -> np.ndarray:
    x = x.clip(0, 1)
    x = x[..., None].repeat(3, -1)
    r = x[..., 0]
    g = x[..., 1]
    b = x[..., 2]
    r = (3 * r - 1).clip(0, 1)
    g = (- np.abs(3 * g - 1.5) + 1.5).clip(0, 1)
    b = (-3 * b + 1).clip(0, 1)
    if type == np.uint8:
        r = (r * 255).astype(np.uint8)
        g = (g * 255).astype(np.uint8)
        b = (b * 255).astype(np.uint8)
    else:
        r = r.astype(type)
        g = g.astype(type)
        b = b.astype(type)
    return np.stack([r, g, b], axis=-1)


if __name__ == '__main__':
    cbar = np.arange(256) / 255.0
    cbar = cmap(cbar[np.newaxis, ...])
    cv2.imwrite(r"D:\files\PHD\myNeRF\paper\Fig\synthetic views\c2.png", cbar)
