import msnerf.PLYLoader as PLYLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from msnerf.tools.miscs import get_cali_matrix

# 定义文件夹路径和要提取的波段
_asd_folder_path = r"D:\files\PHD\myNeRF\data\ASD"  # 替换为你的文件夹路径
_target_wavelengths = [667, 670, 689, 702, 713,
                       730, 741, 754, 769, 781,
                       790, 805, 815, 828, 843,
                       853, 864, 878, 890, 898,
                       911, 919, 929, 939, 949]


def load_asd(ids, mean=True) -> np.ndarray:
    results = []
    for id_ in ids:
        assert isinstance(id_, int)
        ref = pd.read_csv(fr'{_asd_folder_path}\{id_:07d}.asd.txt', sep='\s+', index_col='Wavelength').loc[
            _target_wavelengths].to_numpy().squeeze()
        results.append(ref)
    results = np.stack(results, axis=0)
    if mean:
        results = results.mean(axis=0)
    return results


if __name__ == '__main__':
    pc: np.ndarray = PLYLoader.load_points(
        r"D:\files\PHD\myNeRF\msnerf\outputs\87_0\msnerf\pc_cali.ply")
    pc = pc.astype(np.float32)
    ms = pc[:, -25:]
    ms = ms @ get_cali_matrix().T
    mean = np.mean(ms, axis=0)
    std = np.std(ms, axis=0)
    gt = load_asd(range(115, 120))
    print(mean)
    print(std)
    index = np.argmin(np.sum((gt-ms) ** 2, axis=1))
    fig, ax = plt.subplots()
    # ax.errorbar(_target_wavelengths, mean, yerr=std, label='mean')
    ax.plot(_target_wavelengths, gt, label='ASD gt')
    ax.plot(_target_wavelengths, ms[index], label='optim')
    ax.legend()
    plt.show()
