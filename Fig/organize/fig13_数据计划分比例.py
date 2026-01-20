from pathlib import Path

import PIL.Image as Image
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.colors import Normalize

if __name__ == '__main__':
    base_folder = Path(r"D:\files\PHD\myNeRF\paper\Fig\ratio")
    names = ['不加mask', '只mask', '先后', '交替']
    images = [[], []]
    for name in names:
        images[0].append(np.array(Image.open(base_folder / (name + '.png'))))
        images[1].append(np.array(Image.open(base_folder / (name + '2.png'))))
    col_titles = [
        "Full scene",
        "Object only\n(potted plant)",
        "Sequential training\n(full scene / potted plant)",
        "Interleaved training\n(full scene / potted plant)"
    ]

    plt.rcParams.update({
        'font.size': 12,
    })
    fig = plt.figure(figsize=(12, 4), dpi=300)

    gs = gridspec.GridSpec(2, 5, width_ratios=[1, 1, 1, 1, 0.05], wspace=0.02, hspace=0.02)

    for i in range(8):
        row = i // 4
        col = i % 4
        ax = fig.add_subplot(gs[row, col])
        im = ax.imshow(images[row][col], aspect='equal')
        ax.axis('off')

        if row == 0:
            ax.set_title(col_titles[col], pad=5)

    cbar = plt.colorbar(cm.ScalarMappable(norm=Normalize(vmin=0, vmax=8),
                                          cmap=cm.get_cmap('cividis')),
                        cax=fig.add_subplot(gs[:, 4]),
                        label='Spectral angle (°)')


    plt.subplots_adjust(left=0.01, right=0.95, top=0.85, bottom=0.05)
    plt.savefig(base_folder.parent / "13.pdf", format='pdf', bbox_inches='tight')
