from pathlib import Path

import cv2
import numpy as np
import torch
import tqdm
import yaml
from nerfstudio.engine.trainer import Trainer, TrainerConfig

from msnerf.ms_datamanager import MaskedDataManager
from msnerf.ms_model import MSNerfModel

if __name__ == '__main__':
    path = Path(r"D:\files\PHD\myNeRF\msnerf\outputs\90_0\msnerf\config.yml")
    out_path = path.parent / 'render'
    out_path.mkdir(exist_ok=True)
    config = yaml.load(path.read_text(), Loader=yaml.Loader)
    assert isinstance(config, TrainerConfig)
    config.load_dir = config.get_checkpoint_dir()
    trainer: Trainer = config.setup()
    trainer.setup()
    pipeline = trainer.pipeline
    ms_model: MSNerfModel = trainer.pipeline._model
    datamanager: MaskedDataManager = pipeline.datamanager
    ms_model.eval()
    with torch.no_grad():
        img_num = [355]
        cameras = datamanager.train_dataset.cameras
        for i in tqdm.tqdm(img_num):
            img_idx = i // 10 * 9 + i % 10
            outputs = ms_model.get_outputs_for_reduced_camera(cameras[img_idx])
            img = outputs['ms'].cpu().numpy()
            img = (img.clip(0, 1) * 255.0).astype(np.uint8)
            img_compressed = img.reshape(img.shape[0], img.shape[1],5, 5).transpose(0, 2, 1, 3).reshape(img.shape[0] * 5, img.shape[1] * 5)
            cv2.imwrite(str(out_path / f"{i:04d}.png"), img_compressed)
