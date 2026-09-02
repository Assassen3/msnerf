from pathlib import Path

import matplotlib.pyplot as plt
import msnerf.PLYLoader as PLYLoader

from msnerf.tools.asd import load_asd
from msnerf.tools.miscs import wavelengths

if __name__ == '__main__':
    gts = [load_asd(range(_, _ + 5)) for _ in list(range(110, 145, 5)) + list(range(90, 110, 5))]
    pc = PLYLoader.load_points(r"D:\files\PHD\myNeRF\msnerf\outputs\87_0\msnerf\pc_cali_plant.ply")
    xyz = pc[:, :3]
    mask = (xyz < [0.1668, -0.0809, 0.1948]).all(axis=1) & (xyz > [0.0006, -0.2812, 0.1098]).all(axis=1)
    pc = pc[mask]
    rs = pc[:, 6:6 + 25]

    pc3 = PLYLoader.load_points(r"D:\files\PHD\myNeRF\msnerf\outputs\87_0\msnerf\pc_cali_board.ply")
    rs3 = pc3[:, 6:6 + 25]

    plt.rcParams.update({
        'font.size': 14,
    })
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), dpi=300)

    ax = axes[0]
    ax.plot(wavelengths, gts[0], label='Leaf', color='tab:blue')
    parts = ax.violinplot([rs[:, i] for i in range(25)],
                          positions=wavelengths,
                          widths=[5] * 2 + [10] * 23,
                          showextrema=False,
                          showmeans=False,
                          showmedians=True)
    for part in parts['bodies']:
        part.set_facecolor('tab:orange')
    ax.set_ylim(0, 1)
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Reflectance')
    ax.xaxis.set_ticks([667, 700, 750, 800, 850, 900, 949])

    ax = axes[1]
    parts = ax.violinplot([rs3[:, i] for i in range(25)],
                          positions=wavelengths,
                          widths=[5] * 2 + [10] * 23,
                          showextrema=False,
                          showmeans=False,
                          showmedians=False)
    for part in parts['bodies']:
        part.set_facecolor('tab:orange')
    ax.set_ylim(0, 1)
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Reflectance')
    ax.xaxis.set_ticks([667, 700, 750, 800, 850, 900, 949])
    ax.plot(wavelengths, gts[-1], label='25% reference', color='tab:purple')
    ax.plot(wavelengths, gts[-2] + 0.1, label='40% reference', color='tab:green')
    ax.plot(wavelengths, gts[-4], label='98% reference', color='tab:red')

    plt.tight_layout()
    save_path = Path(r"D:\files\PHD\myNeRF\paper\Fig")
    plt.savefig(save_path / "11.pdf", format='pdf', bbox_inches='tight')
