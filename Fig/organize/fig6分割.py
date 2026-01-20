from pathlib import Path

import PIL.Image as Image
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    base_folder = Path(r"D:\files\PHD\myNeRF\paper\Fig\mask")
    names = ['0176', '0355', '0011', '0011']
    folder = ['0', '3', '1', '2']
    images = []
    for idx, name in enumerate(names):
        for prefix in ['', 'mask_', 'pcall_', 'pc_']:
            img = np.array(Image.open(base_folder / folder[idx] / f"{prefix}{name}.png"))
            if len(img.shape) == 2:
                img = img[:, :, np.newaxis].repeat(3, axis=2)
            images.append(img)
    col_titles = [
        "Raw images",
        "Masked images",
        "Point cloud",
        "Masked point cloud",
    ]

    plt.rcParams.update({
        'font.size': 12,
    })
    fig = plt.figure(figsize=(12, 7), dpi=300)

    gs = gridspec.GridSpec(4, 4, width_ratios=[1, 1, 1, 1], wspace=0, hspace=0)

    for i in range(16):
        row = i // 4
        col = i % 4
        ax = fig.add_subplot(gs[row, col])
        im = ax.imshow(images[i], aspect='equal')
        ax.axis('off')
        if row == 0:
            ax.set_title(col_titles[col])

    plt.subplots_adjust(left=0.02, right=0.95, bottom=0.02, top=0.98)
    plt.tight_layout()
    plt.savefig(base_folder.parent / "6.pdf", format='pdf', bbox_inches='tight')
