import os
from pathlib import Path

import PLYLoader
import numpy as np
import open3d as o3d
import yaml
from nerfstudio.engine.trainer import Trainer

from msnerf.ms_datamanager import MaskedDataManager
from msnerf.ms_model import MSNerfModel
from msnerf.tools.miscs import cmap, get_cali_matrix


def render_pc(datamanager, img_num, xyz, colors, output_path):
    cameras = datamanager.train_dataset.cameras
    dataset_id = img_num // 10 * 9 + img_num % 10
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=2045, height=1085, visible=False)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    vis.add_geometry(pcd)
    ctr = vis.get_view_control()
    params = ctr.convert_to_pinhole_camera_parameters()

    params.intrinsic = o3d.camera.PinholeCameraIntrinsic(width=cameras.width[dataset_id, 0].item(),
                                                         height=cameras.height[dataset_id, 0].item(),
                                                         fx=cameras.fx[dataset_id, 0].item(),
                                                         fy=cameras.fy[dataset_id, 0].item(),
                                                         cx=cameras.cx[dataset_id, 0].item(),
                                                         cy=cameras.cy[dataset_id, 0].item())
    c2w = np.diag([1, 1, 1, 1]).astype(np.float32)
    c2w[:3, :] = cameras.camera_to_worlds[dataset_id].numpy()

    params.extrinsic = np.diag([1, -1, -1, 1]) @ np.linalg.inv(c2w)
    ctr.convert_from_pinhole_camera_parameters(params, allow_arbitrary=True)
    render_op = vis.get_render_option()
    render_op.point_size = 1
    vis.poll_events()
    vis.capture_screen_image(output_path)
    return


if __name__ == '__main__':
    path = Path(r"D:\files\PHD\myNeRF\msnerf\outputs\89_0\msnerf\config.yml")
    img_view = 114
    band = 1
    out_path = Path(r"D:\files\PHD\myNeRF\paper\Fig\pc")
    if not out_path.exists():
        os.makedirs(out_path)
    config = yaml.load(path.read_text(), Loader=yaml.Loader)
    config.load_dir = config.get_checkpoint_dir()
    trainer: Trainer = config.setup()
    trainer.setup()
    pipeline = trainer.pipeline
    ms_model: MSNerfModel = trainer.pipeline._model
    datamanager: MaskedDataManager = pipeline.datamanager

    pc = PLYLoader.load_points(str(path.parent / 'pc_cali_plant.ply'))
    pc[:, 6:31] = pc[:, 6:31]
    colors = pc[:, 6 + band]
    colors = np.clip(colors * 1.5, 0 , 1)
    colors = cmap(colors, np.float64)
    render_pc(datamanager, img_view, pc[:, :3], colors, out_path / f'{img_view:04d}_{band:04d}.png')
