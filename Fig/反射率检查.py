import PLYLoader
import numpy as np
from matplotlib import pyplot as plt

from msnerf.tools.miscs import wavelengths

pc = PLYLoader.load_points(r"D:\files\PHD\myNeRF\msnerf\outputs\87_0\msnerf\pc_cali.ply")
mask = (pc[:, :3] < np.array((0.0780, -0.1057, 0.1593))).all(axis=-1) & (
        pc[:, :3] > np.array((0.0613, -0.1304, 0.1455))).all(axis=-1)

pc = pc[mask]
ms = pc[:, 6:6 + 25]
mean = np.mean(ms, axis=0)

fig = plt.figure()
ax = fig.add_subplot(111)
pc_plot = ax.plot(wavelengths, mean, label='pc')
plt.legend()
plt.show()
