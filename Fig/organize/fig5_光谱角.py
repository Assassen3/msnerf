from pathlib import Path

import PIL.Image as Image
import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.colors import Normalize

if __name__ == '__main__':
    base_folder = Path(r"D:\files\PHD\myNeRF\paper\Fig\sam")
    names = ['87_0176', '90_0355', '61_0011', '83_0012']
    images = []
    for name in names:
        images.append(np.array(Image.open(base_folder / (name + '.png'))))

    plt.rcParams.update({
        'font.size': 12,
    })
    fig = plt.figure(figsize=(12, 6), dpi=300)

    gs = gridspec.GridSpec(2, 3, width_ratios=[1, 1, 0.05], wspace=0, hspace=0)

    for i in range(4):
        row = i // 2
        col = i % 2
        ax = fig.add_subplot(gs[row, col])
        im = ax.imshow(images[i], aspect='equal')
        ax.axis('off')

    plt.colorbar(cm.ScalarMappable(norm=Normalize(vmin=0, vmax=8),
                                   cmap=matplotlib.colormaps['cividis']),
                 cax=fig.add_subplot(gs[:, 2]),
                 label='Spectral angle (°)')

    plt.subplots_adjust(left=0.01, right=0.95, top=0.95, bottom=0.05)
    plt.savefig(base_folder.parent / "5.pdf", format='pdf', bbox_inches='tight')
