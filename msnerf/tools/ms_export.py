from __future__ import annotations

from pathlib import Path
from typing import Optional

import PLYLoader
import numpy as np
import torch
import torch.nn.functional as F
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.data.scene_box import OrientedBox
from nerfstudio.utils.eval_utils import eval_setup


def generate_ms_point_cloud(
        config: Path,
        output_dir: Optional[Path] = None,
        filename: Optional[str] = 'pc.ply',
        num_points: int = 100000,
        obb_center=(0., 0., 0.),
        obb_rotation=(0., 0., 0.),
        obb_scale=(1., 1., 1.),
        off=0.001,
):
    if output_dir is None:
        output_dir = Path(config).parent
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
    _, pipeline, _, _ = eval_setup(config)
    # 这里要手动该batch_size
    crop_obb = OrientedBox.from_params(obb_center, obb_rotation, obb_scale)

    points_done = 0
    points = []
    pixel_ids = []

    offsets = torch.tensor([
        [-off, -off, 0], [-off, 0, 0], [-off, off, 0],
        [0, -off, 0], [0, 0, 0], [0, off, 0],
        [off, -off, 0], [off, 0, 0], [off, off, 0]
    ], dtype=torch.float32, device='cuda')[None, :, :]

    while not points_done >= num_points:
        with torch.no_grad():
            ray_bundle, batch = pipeline.datamanager.next_train(0)

            origins = ray_bundle.origins[:, None, :].expand(-1, 9, -1)
            directions = ray_bundle.directions[:, None, :].expand(-1, 9, -1)
            origins = (origins + torch.cross(directions, offsets, dim=-1)).reshape(-1, 3)
            directions = directions.reshape(-1, 3)
            pixel_area = ray_bundle.pixel_area[:, None, :].expand(-1, 9, -1).reshape(-1, 1)
            camera_indices = ray_bundle.camera_indices[:, None, :].expand(-1, 9, -1).reshape(-1, 1)
            metadata = {'ms_index': ray_bundle.metadata['ms_index'][:, None, :].expand(-1, 9, -1).reshape(-1, 1)}
            ray_bundle = RayBundle(
                origins=origins,
                directions=directions,
                pixel_area=pixel_area,
                camera_indices=camera_indices,
                metadata=metadata,
            )

            outputs = pipeline.model(ray_bundle)

        depth = outputs['depth'].reshape(-1, 9, 1)
        point = origins.reshape(-1, 9, 3) + depth * directions.reshape(-1, 9, 3)
        p_mean = point.mean(dim=1, keepdim=True)
        p_centered = point - p_mean
        C = torch.matmul(p_centered.transpose(1, 2), p_centered)
        eigenvalues, eigenvectors = torch.linalg.eigh(C)  # [B, 3, 3]
        normal = eigenvectors[:, :, 0]  # [B, 3]
        normal = F.normalize(normal, dim=-1)  # [B, 3]

        center_dirs = directions.reshape(-1, 9, 3)[:, 4, :]
        dot = (normal * center_dirs).sum(dim=-1, keepdim=True)  # [B, 1]
        flip_mask = dot > 0
        normal = torch.where(flip_mask, -normal, normal)

        rgba = pipeline.model.get_rgba_image(outputs, 'ms').reshape(-1, 9, 26)[:, 4, :].cpu().numpy()
        depth = depth[:, 4, :].cpu().numpy()
        normal = normal.cpu().numpy()
        ray_bundle = ray_bundle.reshape((-1, 9))[:, 4].to('cpu')

        point = ray_bundle.origins + ray_bundle.directions * depth

        ms = rgba[..., :-1]
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

    points = np.vstack(points)
    pixel_ids = np.vstack(pixel_ids)

    print("Saving Point Cloud...")

    output_ply_path = output_dir / filename
    save_point_cloud_to_ply(points, pixel_ids, output_ply_path)


def save_point_cloud_to_ply(points: np.ndarray, pixel_ids: np.ndarray, output_ply_path: Path):
    name = ['x', 'y', 'z'] + [f'ms_{i}' for i in range(25)] + ['nx', 'ny', 'nz'] + ['img_id', 'pix_pos_y',
                                                                                    'pix_pos_x']
    type_pro = ['float'] * 31 + ['int'] * 3
    pixel_ids = pixel_ids.astype(np.float32)
    PLYLoader.save_points(str(output_ply_path), np.hstack([points, pixel_ids]), name, type_pro)


if __name__ == '__main__':
    generate_ms_point_cloud(config=Path(r"D:\files\PHD\myNeRF\msnerf\outputs\91_0\msnerf\2025-11-12_103447\config.yml"),
                            num_points=100000)
