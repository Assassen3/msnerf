from pathlib import Path

import PIL.Image as Image
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    base_folder = Path(r"D:\files\PHD\myNeRF\paper\Fig\rrpc")
    names = ['87', '89', '61', '83']
    images = [[] for _ in range(3)]
    for name in names:
        images[0].append(np.array(Image.open(base_folder / f'{name}_nx.png'))[:, 75:-75, :])
        images[1].append(np.array(Image.open(base_folder / f'{name}_ny.png'))[:, 75:-75, :])
        images[2].append(np.array(Image.open(base_folder / f'{name}_nz.png'))[:, 75:-75, :])
    col_titles = [
        "Normal vector (X)",
        "Normal vector (Y)",
        "Normal vector (Z)",
    ]

    plt.rcParams.update({
        'font.size': 12,
    })
    fig = plt.figure(figsize=(12, 13), dpi=300)

    gs = gridspec.GridSpec(4, 3, wspace=0.02, hspace=0.02)

    for i in range(12):
        row = i // 3
        col = i % 3
        ax = fig.add_subplot(gs[row, col])
        im = ax.imshow(images[col][row], aspect='equal')
        ax.axis('off')

        if row == 0:
            ax.set_title(col_titles[col])

    plt.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.98)
    plt.tight_layout()
    plt.savefig(base_folder.parent / "8.pdf", format='pdf', bbox_inches='tight')
