from pathlib import Path

import PIL.Image as Image
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    base_folder = Path(r"D:\files\PHD\myNeRF\paper\Fig\synthetic views")
    folder = ['1', '2', '3', '0']
    images = []
    for name in folder:
        for prefix in ['c', 'cc']:
            for suffix in ['gt_single.png', '3DGS.png', 'nerf.jpg', 'ours.png']:
                img = np.array(Image.open(base_folder / name / f"{prefix}_{suffix}"))
                if len(img.shape) == 2:
                    img = img[:, :, np.newaxis].repeat(3, axis=2)
                images.append(img)
    col_titles = [
        "Raw images",
        "3DGS",
        "Nerfacto",
        "Ours",
    ]

    plt.rcParams.update({
        'font.size': 12,
    })
    fig = plt.figure(figsize=(12, 14), dpi=300)

    gs = gridspec.GridSpec(8, 4, wspace=0, hspace=0)

    for i in range(32):
        row = i // 4
        col = i % 4
        ax = fig.add_subplot(gs[row, col])
        im = ax.imshow(images[i], aspect='equal')
        ax.axis('off')
        if row == 0:
            ax.set_title(col_titles[col])

    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    plt.tight_layout()
    plt.savefig(base_folder.parent / "4.pdf", format='pdf', bbox_inches='tight')
