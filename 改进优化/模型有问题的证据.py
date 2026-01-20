import PIL.Image as Image
import numpy as np
from matplotlib import pyplot as plt

from msnerf.tools.miscs import get_cali_matrix

if __name__ == '__main__':
    img = r"D:\files\PHD\myNeRF\data\90_0\images\0345.png"
    img = Image.open(img)
    img = np.array(img)[:1085, :2045]
    img = img.reshape(img.shape[0] // 5, 5, img.shape[1] // 5, 5)
    img = img.transpose((0, 2, 1, 3))
    board = [[1352, 518], [1446, 563], [1505, 626], [1594, 648]]
    leaf = [1256, 841]

    n_idx1 = np.arange(5)[None, None, :, None]
    n_idx2 = np.arange(5)[None, None, None, :]
    board = np.array(board) // 5
    leaf = np.array(leaf) // 5
    board_val = img[board[:, 1][:, None, None, None], board[:, 0][:, None, None, None], n_idx1, n_idx2].reshape(4, 25)
    leaf_val = img[leaf[1:2][:, None, None, None], leaf[0:1][:, None, None, None], n_idx1, n_idx2].reshape(1, 25)
    cali = get_cali_matrix()
    board_val = board_val @ cali.T
    leaf_val = leaf_val @ cali.T
    fig, ax = plt.subplots()
    ax.plot(leaf_val[0] / board_val[0], label='leaf')
    for i in range(4):
        ax.plot(board_val[i] / board_val[0], label=f'board{i}')
    plt.legend()
    plt.show()
