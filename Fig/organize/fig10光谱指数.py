from pathlib import Path

import PIL.Image as Image
import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap, Normalize

if __name__ == '__main__':
    base_folder = Path(r"D:\files\PHD\myNeRF\paper\Fig\vegetable index")
    names = ['cndvigt_', 'cndvir_', 'cndvirpc_']
    images = []
    for name in names:
        for n in ['0131.png', '0176.png']:
            img = np.array(Image.open(base_folder / (name + n)))
            cut_l = img.shape[1] // 3
            cut_r = img.shape[1] // 10
            cut_u = img.shape[0] // 3
            img = img[cut_u:, cut_l:-cut_r, :]
            images.append(img)
    col_titles = [
        "NDVI of raw images",
        "NDVI of generated\nreflectance images",
        "NDVI of reflectance\n point cloud",
    ]

    plt.rcParams.update({
        'font.size': 12,
    })
    fig = plt.figure(figsize=(12, 5.5), dpi=300)

    gs = gridspec.GridSpec(2, 5, width_ratios=[1, 1, 0.05, 1, 0.05], wspace=0.02, hspace=0.02)

    for i in range(6):
        row = i % 2
        col = i // 2

        col_img = col + 1 if col > 1 else col
        ax = fig.add_subplot(gs[row, col_img])
        im = ax.imshow(images[i], aspect='equal')
        ax.axis('off')

        if row == 0:
            ax.set_title(col_titles[col], pad=5)

    plt.colorbar(cm.ScalarMappable(norm=Normalize(vmin=0, vmax=1),
                                   cmap=matplotlib.colormaps['cividis']),
                 cax=fig.add_subplot(gs[:, 2]))

    plt.colorbar(cm.ScalarMappable(norm=Normalize(vmin=0, vmax=1),
                                   cmap=LinearSegmentedColormap.from_list(
                                       "my_colormap",
                                       ['#0000ff', '#00ff00', '#ffff00', '#ff0000']), ),
                 cax=fig.add_subplot(gs[:, 4]))
    plt.subplots_adjust(left=0.01, right=0.95, top=0.85, bottom=0.05)
    plt.savefig(base_folder.parent / "10.pdf", format='pdf', bbox_inches='tight')
