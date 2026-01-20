import cv2
import numpy as np
import tinycudann as tcnn
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from torch.utils.tensorboard import SummaryWriter

from msnerf.tools.miscs import get_cali_matrix

writer = SummaryWriter(log_dir="runs/exp1")
data = cv2.imread(r"D:\files\PHD\myNeRF\data\87_0\images\0000.png", cv2.IMREAD_UNCHANGED)
data = data[:1085, :2045].reshape((217, 5, 409, 5)).transpose((0, 2, 1, 3)).reshape(217, 409, 25)
data = torch.from_numpy(data).half() / 255.0

# mlp = tcnn.NetworkWithInputEncoding(
#     n_input_dims=2,
#     n_output_dims=25,
#     encoding_config={
#         "otype": "Grid",
#         "type": "Hash",
#         "n_levels": 16,
#         "n_features_per_level": 2,
#         "log2_hashmap_size": 14,
#         "base_resolution": 16,
#         "per_level_scale": 1.26,
#         "interpolation": "Linear"
#     },
#     network_config={
#         "otype": "FullyFusedMLP",
#         "activation": "ReLU",
#         "output_activation": "None",
#         "n_neurons": 64,
#         "n_hidden_layers": 3,
#
#     }
# )

encoder = tcnn.Encoding(
    n_input_dims=2,
    encoding_config={
        "otype": "Grid",
        "type": "Hash",
        "n_levels": 16,
        "n_features_per_level": 2,
        "log2_hashmap_size": 14,
        "base_resolution": 16,
        "per_level_scale": 1.26,
        "interpolation": "Linear"
    }
)
mlp = nn.Sequential(encoder,
                    nn.Linear(encoder.n_output_dims, 64, bias=False),
                    nn.ReLU(),
                    # nn.Dropout(p=0.2),
                    nn.Linear(64, 64, bias=False),
                    nn.ReLU(),
                    # nn.Dropout(p=0.2),
                    nn.Conv1d(64, 25, bias=False),
                    )
grad_scaler = GradScaler()
optimizer = torch.optim.Adam(mlp.parameters())
batch_size = 1 << 10
mlp.train()
mlp.cuda()
for epoch in range(1000):
    with torch.no_grad():
        indices_n = torch.rand((batch_size, 2))
        indices = (indices_n * torch.tensor((217, 409))).long()
        y, x = torch.split(indices, split_size_or_sections=1, dim=1)
        gt = data[y, x, :].squeeze()
    optimizer.zero_grad()
    indices_n = indices_n.cuda()
    gt = gt.cuda()
    with autocast(device_type='cuda'):
        pred = mlp(indices_n)
        loss1 = torch.nn.functional.l1_loss(pred, gt)
        loss2 = 1 - torch.mean(torch.nn.functional.cosine_similarity(pred, gt, dim=-1))
        loss = loss1 + loss2 * 0.1

    grad_scaler.scale(loss).backward()
    grad_scaler.step(optimizer)
    grad_scaler.update()
    writer.add_scalar("train/loss_iter", loss1.item(), epoch)
    writer.add_scalar("train/sam", loss2.item(), epoch)
    if epoch %100==0:
        print(loss.item() ** 0.5)
writer.close()

mlp.eval()
with torch.no_grad():
    y = torch.arange(217)
    x = torch.arange(409)
    y, x = torch.meshgrid((y, x), indexing="ij")
    indices = torch.stack([y, x], dim=-1).reshape((-1, 2)).cuda()
    y, x = torch.split(indices.cpu(), split_size_or_sections=1, dim=1)
    gt = data[y, x, :].squeeze().numpy().reshape((217, 409, 25))
    gt = (gt * 255).astype(np.uint8)
    indices = torch.stack([indices[:, 0] / 217.0, indices[:, 1] / 409.0], dim=-1)
    with autocast(device_type='cuda'):
        pred = mlp(indices)
    pred = pred.reshape((217, 409, 25))
    img = pred.cpu().numpy()
    img = (img * 255.0).astype(np.uint8)
for i in range(25):
    cv2.imwrite(fr"C:\Users\EY\Desktop\test\{i}.png", img[..., i])
    cv2.imwrite(fr"C:\Users\EY\Desktop\test\gt_{i}.png", gt[..., i])

cos = np.sum(img.astype(np.float32) * gt.astype(np.float32), axis=-1)
cos = cos / (np.linalg.norm(img, axis=-1) * np.linalg.norm(gt, axis=-1))
cos = np.arccos(cos) * 180 / np.pi
print(np.mean(cos))
plt.imshow(cos)
plt.colorbar()
plt.show()

leaf_id = [1405, 330]
board_id = [958, 521]
a = img[leaf_id[1] // 5, leaf_id[0] // 5] / img[board_id[1] // 5, board_id[0] // 5]
b = gt[leaf_id[1] // 5, leaf_id[0] // 5] / gt[board_id[1] // 5, board_id[0] // 5]
cali = get_cali_matrix().T
a = a @ cali
b = b @ cali
fig, ax = plt.subplots()
ax.plot(a, label='pred')
ax.plot(b, label='gt')
ax.legend()
plt.show()

mask: np.ndarray = cv2.imread(r"D:\files\PHD\myNeRF\data\87_0\masks\0000.png", cv2.IMREAD_UNCHANGED)[:1085, :2045]
mask = mask[2::5, 2::5]
mask = mask == 192
mask = mask.flatten()
cos = np.sum(img.reshape(-1, 25)[mask].astype(np.float32) * gt.reshape(-1, 25)[mask].astype(np.float32), axis=-1)
cos = cos / (np.linalg.norm(img.reshape(-1, 25)[mask], axis=-1) * np.linalg.norm(gt.reshape(-1, 25)[mask], axis=-1))
cos = np.arccos(cos) * 180 / np.pi
print(np.mean(cos))
