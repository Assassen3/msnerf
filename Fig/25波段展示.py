import os
from pathlib import Path

import PLYLoader
import yaml
from matplotlib import pyplot as plt
from nerfstudio.engine.trainer import Trainer, TrainerConfig

from msnerf.ms_datamanager import MaskedDataManager
from msnerf.ms_model import MSNerfModel
from msnerf.tools.render_pc import render_pc

path = Path(r"D:\files\PHD\myNeRF\msnerf\outputs\87_0\msnerf\config.yml")
img_view = 176
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
pc = PLYLoader.load_points(r"D:\files\PHD\myNeRF\msnerf\outputs\87_0\msnerf\pc_cali.ply")
cmap = plt.get_cmap('cividis')
for band in range(25):
    render_pc(datamanager, img_view, pc[:, :3], cmap(pc[:, 6 + band])[:, :3],
              path.parent / 'render' / f'pc_{img_view:04d}_{band:02d}.png')
