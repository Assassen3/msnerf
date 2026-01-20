from __future__ import annotations

import os
from pathlib import Path
from typing import OrderedDict
import json
import cv2
import numpy as np
import torch
from sam2.build_sam import build_sam2_video_predictor
from sam2.sam2_video_predictor import SAM2VideoPredictor

if __name__ == '__main__':
    INPUT_FOLDER = Path(r'D:\files\PHD\myNeRF\data\61_0\images')
    MASKS_FOLDER = Path(r"D:\files\PHD\myNeRF\data\61_0\masks")
    POINTS_JSON_PATH = MASKS_FOLDER.parent / "transforms.json"
    imgs_path = [_ for _ in INPUT_FOLDER.iterdir() if _.suffix == '.png']
    imgs_path.sort()
    h, w = cv2.imread(str(imgs_path[0]), cv2.IMREAD_UNCHANGED).shape

    all_imgs = torch.empty((len(imgs_path), h, w), dtype=torch.float32)
    masks = torch.empty((len(imgs_path), h, w), dtype=torch.uint8)
    for i, img in enumerate(imgs_path):
        img_np = cv2.imread(str(img), cv2.IMREAD_UNCHANGED)
        all_imgs[i] = torch.from_numpy(img_np) / 255.0

    sam2predictor: SAM2VideoPredictor = build_sam2_video_predictor(
        "configs/sam2.1/sam2.1_hiera_s.yaml",
        ckpt_path=r"D:\files\PHD\myNeRF\Grounded-SAM-2\checkpoints\sam2.1_hiera_small.pt", )
    image_size: int = sam2predictor.image_size
    full_images = np.empty((len(imgs_path), image_size, image_size), dtype=np.float32)
    for  i in range(len(imgs_path)):
        full_images[i] = cv2.resize(all_imgs[i].numpy(), (image_size, image_size))
    full_images = torch.from_numpy(full_images)[:, None, :, :].expand((-1, 3, -1, -1))
    state = {
        'images': full_images,
        'num_frames': len(imgs_path),
        'offload_video_to_cpu': False,
        'offload_state_to_cpu': False,
        'video_height': h,
        'video_width': w,
        'device': 'cuda:0',
        'storage_device': torch.device("cuda"),
        'point_inputs_per_obj': {},
        'mask_inputs_per_obj': {},
        'cached_features': {},
        'constants': {},
        'obj_id_to_idx': OrderedDict(),
        'obj_idx_to_id': OrderedDict(),
        'obj_ids': [],
        'output_dict_per_obj': {},
        'temp_output_dict_per_obj': {},
        'frames_tracked_per_obj': {},
    }
    masks = masks.to('cuda')
    with open(POINTS_JSON_PATH, 'r') as f:
        mask_points = json.load(f).get('mask_points', {})
    with torch.no_grad():
        sam2predictor._get_image_feature(state, frame_idx=0, batch_size=1)
        for k, v in {'plant': 192, 'hemisphere': 127, 'hemisphere2': 128, 'board': 64}.items():
            if mask_points.get(k, None) is not None:
                sam2predictor.add_new_points_or_box(state, frame_idx=0, obj_id=v,
                                                    points=torch.tensor(mask_points[k]),
                                                    labels=torch.tensor([1, ] * len(mask_points[k])))
        for out_frame_idx, out_obj_ids, out_mask_logits in sam2predictor.propagate_in_video(state):
            for obj_id, out_obj_id in enumerate(out_obj_ids):
                mask = out_mask_logits[obj_id, 0] > 0.0
                masks[out_frame_idx][mask] = out_obj_id
        masks = masks.to('cpu')
        if not MASKS_FOLDER.exists():
            MASKS_FOLDER.mkdir()
        for i, img_path in enumerate(imgs_path):
            cv2.imwrite(str(MASKS_FOLDER / img_path.name), masks[i].numpy())
