from __future__ import annotations

import os.path
import time
from pathlib import Path

import msnerf.PLYLoader as PLYLoader
import numpy as np
import torch
import tqdm
import yaml
from nerfstudio.engine.trainer import Trainer, TrainerConfig
from nerfstudio.field_components.encodings import SHEncoding
from nerfstudio.field_components.mlp import MLP
from nerfstudio.model_components.losses import MSELoss
from torch import GradScaler, nn
from torch.utils.tensorboard import SummaryWriter

from msnerf.ms_datamanager import MaskedDataManager
from msnerf.ms_model import MSNerfModel
from msnerf.tools.miscs import get_cali_matrix, r2_score, reference_gt_reflectance

if __name__ == '__main__':
    path = Path(r"D:\files\PHD\myNeRF\msnerf\outputs\61_0\msnerf\config.yml")
    config = yaml.load(path.read_text(), Loader=yaml.Loader)
    assert isinstance(config, TrainerConfig)
    config.load_dir = config.get_checkpoint_dir()
    trainer: Trainer = config.setup()
    trainer.setup()
    pipeline = trainer.pipeline
    ms_model: MSNerfModel = trainer.pipeline._model
    datamanager: MaskedDataManager = pipeline.datamanager

    writer = SummaryWriter(str(path.parent / 'run' / str(time.time())))

    r_mlp = MLP(
        in_dim=ms_model.config.geo_feat_dim,
        num_layers=3,
        layer_width=128,
        out_dim=25,
        activation=nn.ReLU(),
        out_activation=nn.Sigmoid(),
        implementation='tcnn',
    )
    direction_encoding = SHEncoding(
        levels=4,
        implementation='tcnn',
    )
    light_mlp = MLP(
        in_dim=2 * direction_encoding.get_out_dim(),
        num_layers=3,
        layer_width=128,
        out_dim=25,
        activation=nn.ReLU(),
        out_activation=nn.Sigmoid(),
        implementation='tcnn',
    )
    # datamanager.update_mask(0, trainer.checkpoint_dir.parent / 'mask')

    # ------光场拟合部分，只训练light_mlp---------
    optimizer = torch.optim.Adam(list(light_mlp.parameters()), lr=1e-2)
    scaler = GradScaler()
    mse_loss = MSELoss()

    r = reference_gt_reflectance
    r = torch.from_numpy(r.squeeze()).to('cuda')
    points = []
    for epoch in tqdm.tqdm(range(100), leave=True, desc='light estimation'):
        optimizer.zero_grad()
        with torch.no_grad():
            ray_bundle, _ = pipeline.datamanager.next_inference(mask_index=127)
            ray_bundle = ms_model.collider(ray_bundle)
            normals = ms_model.get_normals(ray_bundle)
            ray_samples, _, _ = ms_model.proposal_sampler(ray_bundle, density_fns=ms_model.density_fns)
            density, density_embedding = ms_model.field.get_density(ray_samples)
            field_outputs = ms_model.field.get_outputs(ray_samples, density_embedding)
            weights = ray_samples.get_weights(density)
            index = torch.argmax(weights, dim=1, keepdim=True)
            ms = torch.gather(field_outputs['ms'], index=index.expand([-1, -1, 25]), dim=1)
            # density_embedding = torch.gather(density_embedding, index=index.expand([-1, -1, 15]), dim=1)
            # r = r_mlp(density_embedding.view(-1, ms_model.config.geo_feat_dim))
        light_embedding = light_mlp(torch.cat([direction_encoding(ray_bundle.directions.reshape(-1, 3)),
                                               direction_encoding(normals.reshape(-1, 3))], dim=-1))
        loss = mse_loss(r * light_embedding, ms.reshape(-1, 25))
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    if not os.path.exists(path.parent / 'models'):
        os.makedirs(path.parent / 'models')
    torch.save(light_mlp.state_dict(), path.parent / 'models' / 'E.pth')

    # ---------训练新的反射率场---------
    optimizer = torch.optim.Adam(list(r_mlp.parameters()))
    scaler = GradScaler()
    mse_loss = MSELoss()
    for epoch in tqdm.tqdm(range(100), leave=True, desc='r_mlp training'):
        optimizer.zero_grad()
        with torch.no_grad():
            ray_bundle, _ = pipeline.datamanager.next_inference(mask_index=None)
            ray_bundle = ms_model.collider(ray_bundle)
            normals = ms_model.get_normals(ray_bundle)
            with torch.amp.autocast('cuda'):
                ray_samples, _, _ = ms_model.proposal_sampler(ray_bundle, density_fns=ms_model.density_fns)
                density, density_embedding = ms_model.field.get_density(ray_samples)
                field_outputs = ms_model.field.get_outputs(ray_samples, density_embedding)
                weights = ray_samples.get_weights(density)
                index = torch.argmax(weights, dim=1, keepdim=True)
                ms = torch.gather(field_outputs['ms'], index=index.expand([-1, -1, 25]), dim=1)
                density_embedding = torch.gather(density_embedding, index=index.expand([-1, -1, 31]), dim=1)
                light_embedding = light_mlp(torch.cat([direction_encoding(ray_bundle.directions.reshape(-1, 3)),
                                                       direction_encoding(normals.reshape(-1, 3))], dim=-1))
        with torch.amp.autocast('cuda'):
            r = r_mlp(density_embedding.view(-1, ms_model.config.geo_feat_dim))
            loss = mse_loss(r * light_embedding, ms.reshape(-1, 25))
            mse_val = loss.detach()
            writer.add_scalar('r_loss', mse_val, epoch)
            r2 = r2_score(light_embedding * r, ms.reshape(-1, 25))
            writer.add_scalar('r_r2', r2, epoch)
            # 计算MAE和RMSE
            mae = torch.mean(torch.abs(r * light_embedding - ms.reshape(-1, 25)))
            rmse = torch.sqrt(mse_val)
            writer.add_scalar('r_mae', mae, epoch)
            writer.add_scalar('r_rmse', rmse, epoch)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        # if epoch % 10 == 0:
        #     tqdm.tqdm.write(f'{str(mse_val.item() ** 0.5)}      {str(r2.item())}    {str(torch.mean(r).item())}')
    torch.save(r_mlp.state_dict(), path.parent / 'models' / 'R.pth')

    # -------导出点云---------
    cali = get_cali_matrix().T
    points = []
    indices = []
    mask_list = [192] * 6 + [128] * 2 + [127] * 2
    with torch.no_grad():
        for epoch in tqdm.tqdm(range(len(mask_list)), leave=True):
            ray_bundle, batch = pipeline.datamanager.next_inference(mask_index=mask_list[epoch])
            ray_bundle = ms_model.collider(ray_bundle)
            normals = ms_model.get_normals(ray_bundle)
            indices.append(batch['indices'].cpu().numpy())
            with torch.amp.autocast('cuda'):
                ray_samples, _, _ = ms_model.proposal_sampler(ray_bundle, density_fns=ms_model.density_fns)
                density, density_embedding = ms_model.field.get_density(ray_samples)
                weights = ray_samples.get_weights(density)
                index = torch.argmax(weights, dim=1, keepdim=True)
                density_embedding = torch.gather(density_embedding, dim=1,
                                                 index=index.expand((-1, -1, ms_model.config.geo_feat_dim)))
                r = r_mlp(density_embedding.reshape(-1, ms_model.config.geo_feat_dim))
                point = ray_samples.frustums.get_positions()
                point = torch.gather(point, dim=1, index=index.expand((-1, -1, 3)))
                points.append(np.hstack([point.reshape(-1, 3).cpu().numpy(),
                                         normals.reshape(-1, 3).cpu().numpy(),
                                         r.reshape(-1, 25).cpu().numpy() @ cali,
                                         ray_bundle.directions.reshape(-1, 3).cpu().numpy(), ]))
    points = np.vstack(points)
    indices = np.vstack(indices).astype(np.float32)
    points = np.hstack([points, indices])
    # 过滤
    xyz = points[:, :3]
    mask3 = (xyz < 0.5) & (xyz > -0.5)
    mask = mask3.all(axis=1)
    points = points[mask]
    name = ['x', 'y', 'z'] + ['nx', 'ny', 'nz'] + [f'ms_{i}' for i in range(25)] + ['dirx', 'diry', 'dirz'] + [
        'img_idx', 'y', 'x']
    type_pro = ['float'] * 34 + ['int'] * 3
    PLYLoader.save_points(str(path.parent / 'pc_cali.ply'), points, name,
                          type_pro)

    points = []
    indices = []
    mask_list = [255] * 10
    with torch.no_grad():
        for epoch in tqdm.tqdm(range(len(mask_list)), leave=True):
            ray_bundle, batch = pipeline.datamanager.next_inference(mask_index=mask_list[epoch])
            ray_bundle = ms_model.collider(ray_bundle)
            normals = ms_model.get_normals(ray_bundle)
            indices.append(batch['indices'].cpu().numpy())
            with torch.amp.autocast('cuda'):
                ray_samples, _, _ = ms_model.proposal_sampler(ray_bundle, density_fns=ms_model.density_fns)
                density, density_embedding = ms_model.field.get_density(ray_samples)
                weights = ray_samples.get_weights(density)
                index = torch.argmax(weights, dim=1, keepdim=True)
                density_embedding = torch.gather(density_embedding, dim=1,
                                                 index=index.expand((-1, -1, ms_model.config.geo_feat_dim)))
                r = r_mlp(density_embedding.reshape(-1, ms_model.config.geo_feat_dim))
                point = ray_samples.frustums.get_positions()
                point = torch.gather(point, dim=1, index=index.expand((-1, -1, 3)))
                points.append(np.hstack([point.reshape(-1, 3).cpu().numpy(),
                                         normals.reshape(-1, 3).cpu().numpy(),
                                         r.reshape(-1, 25).cpu().numpy() @ cali,
                                         ray_bundle.directions.reshape(-1, 3).cpu().numpy(), ]))
    points = np.vstack(points)
    indices = np.vstack(indices).astype(np.float32)
    points = np.hstack([points, indices])
    # 过滤
    xyz = points[:, :3]
    mask3 = (xyz < 0.5) & (xyz > -0.5)
    mask = mask3.all(axis=1)
    points = points[mask]
    name = ['x', 'y', 'z'] + ['nx', 'ny', 'nz'] + [f'ms_{i}' for i in range(25)] + ['dirx', 'diry', 'dirz'] + [
        'img_idx', 'y', 'x']
    type_pro = ['float'] * 34 + ['int'] * 3
    PLYLoader.save_points(str(path.parent / 'pc_cali_plant.ply'), points, name,
                          type_pro)

