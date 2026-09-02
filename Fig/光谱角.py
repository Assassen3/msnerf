import PIL.Image as Image
import numpy as np
from matplotlib import pyplot as plt


if __name__ == '__main__':
    datasets = [61, 83, 87, 90]
    img_nums = [11, 12, 176, 355]

    for i in range(len(datasets)):
        dataset = datasets[i]
        img_num = img_nums[i]

        gt_file = rf"D:\files\PHD\myNeRF\data\{dataset}_0\images\{img_num:04d}.png"
        gt = np.array(Image.open(gt_file))[:1085, :2045] / 255.0
        gt = gt.reshape(217, 5, 409, 5).transpose((0, 2, 1, 3)).reshape(217, 409, 25)

        pred_file = fr"D:\files\PHD\myNeRF\msnerf\outputs\{dataset}_0\msnerf\render\{img_num:04d}.png"
        pred = np.array(Image.open(pred_file))[:1085, :2045] / 255.0
        pred = pred.reshape(217, 5, 409, 5).transpose((0, 2, 1, 3)).reshape(217, 409, 25)

        dot = np.sum(pred * gt, axis=-1)
        norm_ms = np.linalg.norm(pred, axis=-1)
        norm_img = np.linalg.norm(gt, axis=-1)
        cos = dot / (norm_ms * norm_img)
        cos = np.degrees(np.arccos(cos))
        normalized_cos = np.clip(cos, 0, 8) / 8.0
        cmap = plt.get_cmap('cividis')
        normalized_cos = cmap(normalized_cos)
        Image.fromarray((normalized_cos *255).astype(np.uint8)).save(rf'D:\files\PHD\myNeRF\paper\Fig\sam\{dataset}_{img_num:04d}.png')

