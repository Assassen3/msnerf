from pathlib import Path

import PIL.Image as Image
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap, Normalize

if __name__ == '__main__':
    base_folder = Path(r"D:\files\PHD\myNeRF\paper\Fig\pc")
    names = ['87', '89', '61', '83']
    images = []
    for name in names:
        for band in ['0', '1', '13', '24']:
            img = np.array(Image.open(base_folder / f"{name}_{band}.png"))
            images.append(img)
    col_titles = [
        "667 nm",
        "670 nm",
        "843 nm",
        "949 nm"
    ]

    plt.rcParams.update({
        'font.size': 12,
    })
    fig = plt.figure(figsize=(12, 9), dpi=300)

    gs = gridspec.GridSpec(4, 5, width_ratios=[1, 1, 1, 1, 0.05], wspace=0, hspace=0)

    for i in range(16):
        row = i // 4
        col = i % 4
        ax = fig.add_subplot(gs[row, col])
        im = ax.imshow(images[i], aspect='equal')
        ax.axis('off')
        if row == 0:
            ax.set_title(col_titles[col])

    cbar = plt.colorbar(cm.ScalarMappable(norm=Normalize(vmin=0, vmax=1),
                                          cmap=LinearSegmentedColormap.from_list(
                                              "my_colormap",
                                              ['#0000ff', '#00ff00', '#ffff00', '#ff0000']), ),
                        cax=fig.add_subplot(gs[1:3, 4]),
                        label='Reflection')

    plt.subplots_adjust(left=0.02, right=0.95, bottom=0.02, top=0.98)
    plt.tight_layout()
    plt.savefig(base_folder.parent / "7.pdf", format='pdf', bbox_inches='tight')
