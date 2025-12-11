import os
from pathlib import Path

import PLYLoader
import cv2
import numpy as np
import torch
import tqdm
import yaml
from nerfstudio.engine.trainer import Trainer, TrainerConfig

from msnerf.ms_datamanager import MaskedDataManager
from msnerf.ms_model import MSNerfModel

if __name__ == '__main__':
    path = Path(r"D:\files\PHD\myNeRF\msnerf\outputs\89_0\msnerf\config.yml")
    out_path = path.parent / 'render'
    if not out_path.exists():
        os.makedirs(out_path)
    config = yaml.load(path.read_text(), Loader=yaml.Loader)
    assert isinstance(config, TrainerConfig)
    config.load_dir = config.get_checkpoint_dir()
    trainer: Trainer = config.setup()
    trainer.setup()
    pipeline = trainer.pipeline
    ms_model: MSNerfModel = trainer.pipeline._model
    datamanager: MaskedDataManager = pipeline.datamanager
    with torch.no_grad():
        img_num = [355]
        cameras = datamanager.train_dataset.cameras
        for i in tqdm.tqdm(img_num):
            img_idx = i // 10 * 9 + i % 10
            outputs = ms_model.get_outputs_for_camera(cameras[img_idx])
            img = outputs['ms'].cpu().numpy()
            img = (img.clip(0, 1) * 255.0).astype(np.uint8)
            for band in range(25):
                cv2.imwrite(str(out_path / f'{i:04d}_{band}.png'), img[..., band])
