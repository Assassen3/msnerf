import math

import PIL.Image as Image
import numpy as np
from matplotlib import pyplot as plt

from msnerf.tools.miscs import get_cali_matrix

if __name__ == '__main__':
    img = r"D:\files\PHD\myNeRF\data\83_0\images\0000.png"
    ms_img = np.empty((1085, 2045, 25), dtype=np.uint8)
    for band in range(25):
        ms_img[..., band] = np.array(Image.open(fr"D:\files\PHD\myNeRF\msnerf\outputs\83_0\msnerf\render\0000_{band}.png"))[:1085, :2045]
    img = Image.open(img)
    img = np.array(img)[:1085, :2045]
    img = img.reshape(img.shape[0] // 5, 5, img.shape[1] // 5, 5)
    img = img.transpose((0, 2, 1, 3))
    board_id = [[1012, 840], [1128, 844], [1203, 899], [1280, 901]]
    leaf_id = [[1106, 127],]

    n_idx1 = np.arange(5)[None, None, :, None]
    n_idx2 = np.arange(5)[None, None, None, :]
    board = np.array(board_id) // 5
    leaf = np.array(leaf_id) // 5
    board_val = img[board[:, 1][:, None, None, None], board[:, 0][:, None, None, None], n_idx1, n_idx2].reshape(4, 25)
    leaf_val = img[leaf[:, 1][:, None, None, None], leaf[:, 0][:, None, None, None], n_idx1, n_idx2].reshape(1, 25)
    n_idx3 = np.arange(25)[None, None, :]
    board_val_ms = ms_img[np.array(board_id)[:, 1][:, None, None], np.array(board_id)[:, 0][:, None, None], n_idx3].reshape(4, 25)
    leaf_val_ms = ms_img[np.array(leaf_id)[:, 1][:, None, None], np.array(leaf_id)[:, 0][:, None, None], n_idx3].reshape(1, 25)

    cali = get_cali_matrix()
    board_val = board_val @ cali.T
    leaf_val = leaf_val @ cali.T
    # board_val_ms = board_val_ms @ cali.T
    # leaf_val_ms = leaf_val_ms @ cali.T
    fig, ax = plt.subplots()
    ax.plot(leaf_val[0] / board_val[2], label='leaf')
    ax.plot(leaf_val_ms[0] / board_val_ms[2], label='ms/leaf')
    for i in range(4):
        ax.plot(board_val[i] / board_val[2], label=f'board{i}')
        ax.plot(board_val_ms[i] / board_val_ms[2], label=f'ms/board{i}')
    plt.legend()
    plt.show()

    dot = np.sum(leaf_val_ms[0] * leaf_val[0])
    norm_ms = np.linalg.norm(leaf_val_ms[0])
    norm_img = np.linalg.norm(leaf_val[0])
    cos = dot / (norm_ms * norm_img)
    print(math.acos(cos))