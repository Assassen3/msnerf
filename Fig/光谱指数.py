from pathlib import Path

import PLYLoader
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from nerfstudio.cameras.rays import RayBundle

from msnerf.tools.miscs import cmap as cmap2
from msnerf.tools.miscs import get_cali_matrix
from msnerf.tools.render_pc import render_pc


def get_outputs_for_camera_ray_bundle(model, r_model, camera_ray_bundle: RayBundle):
    num_rays_per_chunk = model.config.eval_num_rays_per_chunk
    image_height, image_width = camera_ray_bundle.origins.shape[:2]
    num_rays = len(camera_ray_bundle)
    outputs_lists = []
    for i in range(0, num_rays, num_rays_per_chunk):
        start_idx = i
        end_idx = i + num_rays_per_chunk
        ray_bundle = camera_ray_bundle.get_row_major_sliced_ray_bundle(start_idx, end_idx)
        ray_bundle = ray_bundle.to(model.device)
        ray_bundle = ms_model.collider(ray_bundle)

        with torch.amp.autocast('cuda'):
            ray_samples, _, _ = ms_model.proposal_sampler(ray_bundle, density_fns=ms_model.density_fns)
            density, density_embedding = ms_model.field.get_density(ray_samples)
            weights = ray_samples.get_weights(density)
            index = torch.argmax(weights, dim=1, keepdim=True)
            density_embedding = torch.gather(density_embedding, dim=1,
                                             index=index.expand((-1, -1, ms_model.config.geo_feat_dim)))
            r = r_model(density_embedding.reshape(-1, ms_model.config.geo_feat_dim))
            outputs_lists.append(r)
    outputs = torch.cat(outputs_lists).reshape(image_height, image_width, -1)
    return outputs


if __name__ == '__main__':
    dst_dir = Path(r"D:\files\PHD\myNeRF\paper\Fig\vegetable index")
    dataset = 87
    img_num = 176

    path = Path(rf"D:\files\PHD\myNeRF\msnerf\outputs\{dataset}_0\msnerf\config.yml")
    config = yaml.load(path.read_text(), Loader=yaml.Loader)
    config.load_dir = config.get_checkpoint_dir()
    trainer = config.setup()
    trainer.setup()
    pipeline = trainer.pipeline
    ms_model = trainer.pipeline._model
    datamanager = pipeline.datamanager

    # 真值图片
    img_path = Path(rf"D:\files\PHD\myNeRF\data\{dataset}_0\images") / f"{img_num:04d}.png"
    cali = get_cali_matrix().T
    img = (cv2.imread(img_path, cv2.IMREAD_UNCHANGED)[:1085, :2045]
           .reshape((217, 5, 409, 5)).transpose(0, 2, 1, 3).reshape((217, 409, 25))) @ cali
    ndvi = ((img[..., 14] - img[..., 0]) / (img[..., 0] + img[..., 14]) * 255).astype(np.uint8)
    cv2.imwrite(str(dst_dir / f"ndvigt_{img_num:04d}.png"), ndvi)
    cmap = plt.get_cmap('cividis')
    ndvi = cv2.cvtColor((cmap(ndvi) * 255).astype(np.uint8), cv2.COLOR_BGRA2RGB)
    cv2.imwrite(str(dst_dir / f"cndvigt_{img_num:04d}.png"), ndvi)

    # with torch.no_grad():
    #     cameras = datamanager.train_dataset.cameras
    #     img_idx = img_num // 10 * 9 + img_num % 10
    #     camera = cameras[img_idx]
    #     ray_bundle = camera.generate_rays(camera_indices=0, keep_shape=True)
    #     ms_per_row = int(camera.metadata['num_ms'] ** 0.5)
    #     ms_index = (torch.arange(ray_bundle.shape[1], device=camera.device)[None, :] % ms_per_row + (
    #             torch.arange(ray_bundle.shape[0], device=camera.device)[:, None] % ms_per_row) * ms_per_row)
    #     ray_bundle.metadata['ms_index'] = ms_index[..., None]
    #     r_model = MLP(
    #         in_dim=ms_model.config.geo_feat_dim,
    #         num_layers=3,
    #         layer_width=128,
    #         out_dim=25,
    #         activation=nn.ReLU(),
    #         out_activation=nn.Sigmoid(),
    #         implementation='tcnn',
    #     )
    #     state_dict = torch.load(path.parent / 'models' / 'R.pth', map_location="cuda")
    #     r_model.cuda().load_state_dict(state_dict)
    #     r = get_outputs_for_camera_ray_bundle(ms_model, r_model, ray_bundle)
    #     r = r.cpu().numpy()[:1085, :2045] @ cali
    # ndvi_r = (r[..., 14] - r[..., 0]) / (r[..., 0] + r[..., 14]) * 255
    # ndvi_r = np.clip(ndvi_r, 0, 255).astype(np.uint8)
    # cv2.imwrite(str(dst_dir / f"ndvir_{img_num:04d}.png"), ndvi_r)
    # ndvi_r = cv2.cvtColor((cmap(ndvi_r) * 255).astype(np.uint8), cv2.COLOR_BGRA2RGB)
    # cv2.imwrite(str(dst_dir / f"cndvir_{img_num:04d}.png"), ndvi_r)

    pc = PLYLoader.load_points(str(path.parent / 'pc_cali.ply'))
    r = pc[:, 6:6 + 25]
    r = (r[:, 14] - r[:, 0]) / (r[:, 0] + r[:, 14])
    xyz = pc[:, :3]
    render_pc(datamanager, img_num, xyz, r[:, None].repeat(3, -1), dst_dir / f"ndvirpc_{img_num:04d}.png")
    c_r = cmap2(r, type=np.float64)
    render_pc(datamanager, img_num, xyz, c_r, dst_dir / f"cndvirpc_{img_num:04d}.png")
