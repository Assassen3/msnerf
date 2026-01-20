from pathlib import Path

import PLYLoader
import matplotlib.gridspec as gridspec
import numpy as np
import torch
import yaml
from matplotlib import pyplot as plt
from nerfstudio.field_components import MLP
from nerfstudio.field_components.encodings import SHEncoding
from torch import nn

if __name__ == '__main__':
    path = Path(r"D:\files\PHD\myNeRF\msnerf\outputs\89_0\msnerf\config.yml")
    config = yaml.load(path.read_text(), Loader=yaml.Loader)
    config.load_dir = config.get_checkpoint_dir()
    trainer = config.setup()
    trainer.setup()
    pipeline = trainer.pipeline
    ms_model = trainer.pipeline._model
    datamanager = pipeline.datamanager

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
    light_mlp.load_state_dict(torch.load(path.parent / 'models' / 'E.pth'))
    light_mlp.eval()

    fig = plt.figure(figsize=(12, 5), dpi=180)
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 0.05])
    ax1 = fig.add_subplot(gs[0], polar=True)
    ax2 = fig.add_subplot(gs[1], polar=True)

    # 从光场估计MLP中绘制变化图
    n_phi, n_theta = 50, 50
    phi = np.linspace(0, 2 * np.pi, n_phi)  # 方位角 0~2pi
    theta = np.linspace(0, np.pi / 2, n_theta)  # 天顶角 0~pi/2
    phi_grid, theta_grid = np.meshgrid(phi, theta)
    normals = np.stack([
        np.cos(phi_grid) * np.sin(theta_grid),
        np.sin(phi_grid) * np.sin(theta_grid),
        np.cos(theta_grid)
    ], axis=-1).reshape(-1, 3)
    out_dir = np.array([-1, -1, -1])
    out_dir = out_dir / np.linalg.norm(out_dir)
    out_dirs = np.tile(out_dir, (normals.shape[0], 1))
    X1 = torch.from_numpy(out_dirs).cuda()
    X2 = torch.from_numpy(normals).cuda()
    X = torch.cat([direction_encoding(X1), direction_encoding(X2)], dim=-1)
    Y = light_mlp(X).cpu().detach().numpy()
    cout = ax1.contourf(phi_grid, theta_grid, Y[:, 24].reshape(50, 50), 100, cmap='YlGn')
    # ax1.axis('off')

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
    normals = normals.cpu().detach().numpy()
    ms = ms.cpu().detach().numpy()[:, 0, 24]
    theta = np.arctan2(normals[:, 1], normals[:, 0])
    phi = np.arccos(normals[:, 2])
    mask = normals[:, 2]>0
    theta = theta[mask][:4000]
    phi = phi[mask][:4000]
    ms = ms[mask][:4000]
    scat = ax2.scatter(theta, phi, c=ms, cmap='YlGn', s=1)
    fig.colorbar(cout, ax=ax1)
    fig.colorbar(scat, ax=ax2)
    plt.show()
