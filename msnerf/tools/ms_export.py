from __future__ import annotations

from pathlib import Path
from typing import Optional

import PLYLoader
import numpy as np
import torch
from nerfstudio.data.scene_box import OrientedBox
from nerfstudio.pipelines.base_pipeline import Pipeline
from tqdm import tqdm


def generate_ms_point_cloud(
        pipeline: Pipeline,
        output_dir: Path,
        filename: Optional[str] = 'pc.ply',
        num_points: int = 100000,
        obb_center=(0., 0., 0.),
        obb_rotation=(0., 0., 0.),
        obb_scale=(1., 1., 1.),
        mask_index: Optional[int] = 255,
):
    crop_obb = OrientedBox.from_params(obb_center, obb_rotation, obb_scale)

    points_done = 0
    points = []
    pixel_ids = []

    bar = tqdm(total=num_points, leave=True, desc='Generating point cloud')
    while not points_done >= num_points:
        with torch.no_grad():
            ray_bundle, batch = pipeline.datamanager.next_inference(mask_index=mask_index)
            outputs = pipeline.model(ray_bundle)
            normal = pipeline.model.get_normals(ray_bundle)
            depth = outputs['depth'].to('cpu')

            ms = outputs['ms'].cpu().numpy()
            normal = normal.cpu().numpy()
            ray_bundle = ray_bundle.to('cpu')
            point = ray_bundle.origins + ray_bundle.directions * depth

            mask = crop_obb.within(point).numpy()
            pixel_id = batch['indices']
            point = point[mask]
            ms = ms[mask]
            normal = normal[mask]
            point = np.hstack([point, ms, normal])
            pixel_id = pixel_id[mask]

            points.append(point)
            pixel_ids.append(pixel_id)
            points_done += point.shape[0]
            bar.update(point.shape[0])

    points = np.vstack(points)
    pixel_ids = np.vstack(pixel_ids)

    output_ply_path = output_dir / filename
    save_point_cloud_to_ply(points, pixel_ids, output_ply_path)


def save_point_cloud_to_ply(points: np.ndarray, pixel_ids: np.ndarray, output_ply_path: Path):
    name = ['x', 'y', 'z'] + [f'ms_{i}' for i in range(25)] + ['nx', 'ny', 'nz'] + ['img_id', 'pix_pos_y',
                                                                                    'pix_pos_x']
    type_pro = ['float'] * 31 + ['int'] * 3
    pixel_ids = pixel_ids.astype(np.float32)
    PLYLoader.save_points(str(output_ply_path), np.hstack([points, pixel_ids]), name, type_pro)


if __name__ == '__main__':
    import yaml
    from nerfstudio.engine.trainer import Trainer, TrainerConfig
    from msnerf.ms_model import MSNerfModel

    from msnerf.ms_datamanager import MaskedDataManager

    path = Path(r"D:\files\PHD\myNeRF\msnerf\outputs\89_0\msnerf\config.yml")
    config = yaml.load(path.read_text(), Loader=yaml.Loader)
    assert isinstance(config, TrainerConfig)
    config.load_dir = config.get_checkpoint_dir()
    trainer: Trainer = config.setup()
    trainer.setup()
    pipeline = trainer.pipeline
    ms_model: MSNerfModel = trainer.pipeline._model
    datamanager: MaskedDataManager = pipeline.datamanager
    generate_ms_point_cloud(pipeline=pipeline,
                            output_dir=path.parent,
                            filename='pc.ply',
                            num_points=100000,
                            mask_index=255)
