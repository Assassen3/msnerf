from pathlib import Path

import PIL.Image as Image
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap, Normalize

if __name__ == '__main__':
    base_folder = Path(r"D:\files\PHD\myNeRF\paper\Fig\rrpc")
    names = ['87', '89', '61', '83']
    images = [[] for _ in range(3)]
    for name in names:
        images[0].append(np.array(Image.open(base_folder / f'{name}_ori.png'))[:, 75:-75, :])
        images[1].append(np.array(Image.open(base_folder / f'{name}_cali.png'))[:, 75:-75, :])

    plt.rcParams.update({
        'font.size': 12,
    })
    fig = plt.figure(figsize=(12, 5), dpi=300)

    gs = gridspec.GridSpec(2, 5, width_ratios=[1, 1, 1, 1, 0.05], wspace=0, hspace=0.07)

    for i in range(8):
        row = i // 4
        col = i % 4
        ax = fig.add_subplot(gs[row, col])
        im = ax.imshow(images[row][col], aspect='equal')
        ax.axis('off')

    cbar = plt.colorbar(cm.ScalarMappable(norm=Normalize(vmin=0, vmax=1),
                                          cmap=LinearSegmentedColormap.from_list(
                                              "my_colormap",
                                              ['#0000ff', '#00ff00', '#ffff00', '#ff0000']), ),
                        cax=fig.add_subplot(gs[0, 4]),
                        label='Radiance',)

    cbar = plt.colorbar(cm.ScalarMappable(norm=Normalize(vmin=0, vmax=1),
                                          cmap=LinearSegmentedColormap.from_list(
                                              "my_colormap",
                                              ['#0000ff', '#00ff00', '#ffff00', '#ff0000']), ),
                        cax=fig.add_subplot(gs[1, 4]),
                        label='Reflection')

    plt.subplots_adjust(left=0.02, right=0.95, bottom=0.02, top=0.98)
    plt.tight_layout()
    plt.savefig(base_folder.parent / "9.pdf", format='pdf', bbox_inches='tight')
