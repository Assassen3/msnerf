from __future__ import annotations

import concurrent
import os
from concurrent.futures import as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Literal, Optional, OrderedDict, Tuple, Type, Union

import cv2
import numpy as np
import torch
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.configs.dataparser_configs import AnnotatedDataParserUnion
from nerfstudio.data.datamanagers.base_datamanager import DataManager, DataManagerConfig
from nerfstudio.data.utils.dataparsers_utils import get_train_eval_split_fraction
from nerfstudio.model_components.ray_generators import RayGenerator
from rich.progress import track
from sam2.build_sam import build_sam2_video_predictor
from sam2.sam2_video_predictor import SAM2VideoPredictor
from torch.nn.parameter import Parameter

from msnerf.ms_dataparser import MSDataParserConfig
from msnerf.ms_dataset import MSSRDataset


class MSRayGenerator(RayGenerator):
    def forward(self, ray_indices):
        ray_bundle = super().forward(ray_indices)
        y = ray_indices[:, 1]  # row indices
        x = ray_indices[:, 2]  # col indices
        assert "num_ms" in self.cameras.metadata
        ms_per_row = int(self.cameras.metadata["num_ms"] ** 0.5)
        ray_bundle.metadata["ms_index"] = ((x % ms_per_row + y % ms_per_row * ms_per_row)
                                           .unsqueeze(-1).to(ray_bundle.camera_indices.device).to(torch.int64))
        return ray_bundle


class EvalLoader():
    def __init__(self, datasets, data):
        self.datasets = datasets
        self.data = data
        self.num = self.data.shape[0]
        self._i = 0

    def __iter__(self):
        return self

    def __next__(self):
        self._i += 1
        if self._i >= self.num:
            raise StopIteration
        return self.datasets.cameras[self._i], {'image': self.data[self._i]}

    def __len__(self):
        return self.num


@dataclass
class MaskedDataManagerConfig(DataManagerConfig):
    _target: Type = field(default_factory=lambda: MaskedDataManager)
    dataparser: AnnotatedDataParserUnion = field(default_factory=MSDataParserConfig)
    train_num_rays_per_batch: int = 1 << 14
    eval_num_rays_per_batch: int = 1 << 14
    sam2_ckpt_path: Optional[str] = None


class MaskedDataManager(DataManager):

    def __init__(self, config: MaskedDataManagerConfig,
                 device: Union[torch.device, str] = "cpu",
                 test_mode: Literal["test", "val", "inference"] = "val",
                 **kwargs):

        self.config = config
        self.device = device
        self.test_mode = test_mode
        self.dataparser = self.config.dataparser.setup()
        self.config.dataparser.data = Path(self.config.data)
        self.train_dataset = MSSRDataset(self.dataparser.get_dataparser_outputs(split="train"))
        self.eval_dataset = MSSRDataset(self.dataparser.get_dataparser_outputs(split="val"))
        self.masks = None
        # self.cali = torch.from_numpy(get_cali_matrix().T)
        super().__init__()
        self.setup_masks()

    def setup_train(self):
        # Data
        num_train_data = len(self.train_dataset)
        h, w = self.train_dataset.cameras.height[0, 0].item(), self.train_dataset.cameras.width[0, 0].item()
        self.train_data = torch.empty((num_train_data, h, w), dtype=torch.float32, device='cpu')

        def load_data(dataset, idx, dst):
            dst[idx] = dataset[idx]['image'].reshape(h, w)

        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
            futures = [executor.submit(load_data, self.train_dataset, idx, self.train_data) for idx in
                       range(num_train_data)]
            for f in track(as_completed(futures), total=num_train_data, description="Loading train data",
                           transient=True):
                f.result()
        # Ray generator
        self.train_ray_generator = MSRayGenerator(self.train_dataset.cameras.to(self.device))

    def setup_eval(self):
        # Data
        num_eval_data = len(self.eval_dataset)
        h, w = self.eval_dataset.cameras.height[0].item(), self.eval_dataset.cameras.width[0].item()
        self.eval_data = torch.empty((num_eval_data, h, w), dtype=torch.float32, device='cpu')

        def load_data(dataset, idx, dst):
            dst[idx] = dataset[idx]['image'].reshape(h, w)

        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
            futures = [executor.submit(load_data, self.eval_dataset, idx, self.eval_data) for idx in
                       range(num_eval_data)]
            for f in track(as_completed(futures), total=num_eval_data, description="Loading eval data", transient=True):
                f.result()
        # Ray generator
        self.eval_ray_generator = MSRayGenerator(self.eval_dataset.cameras.to(self.device))
        self.fixed_indices_eval_dataloader = EvalLoader(self.eval_dataset, self.eval_data)

    def setup_masks(self):
        all_datasets_num = len(self.train_dataset) + len(self.eval_dataset)
        h, w = self.train_dataset.cameras.height[0, 0].item(), self.train_dataset.cameras.width[0, 0].item()
        self.masks = torch.empty((all_datasets_num, h, w), dtype=torch.uint8, device='cpu')
        i_train, i_eval = get_train_eval_split_fraction(list(range(all_datasets_num)),
                                                        self.config.dataparser.train_split_fraction)
        # Already have masks, just read
        if self.train_dataset._dataparser_outputs.mask_filenames is not None:
            mask_filenames = self.train_dataset._dataparser_outputs.mask_filenames
        elif (self.config.data.parent / 'masks').exists():
            mask_filenames = [self.config.data.parent / 'masks' / x.name for x in
                              self.train_dataset._dataparser_outputs.image_filenames + self.eval_dataset._dataparser_outputs.image_filenames]
            mask_filenames.sort()
        # Don't have masks, use sam2
        else:
            sam2predictor: SAM2VideoPredictor = build_sam2_video_predictor(
                "configs/sam2.1/sam2.1_hiera_s.yaml",
                ckpt_path=self.config.sam2_ckpt_path)
            image_size: int = sam2predictor.image_size
            full_images = np.empty((all_datasets_num, image_size, image_size), dtype=np.float32)
            for idx, i in enumerate(i_train):
                full_images[i, ...] = cv2.resize(self.train_data[idx].numpy(), (image_size, image_size))
            for idx, i in enumerate(i_eval):
                full_images[i, ...] = cv2.resize(self.eval_data[idx].numpy(), (image_size, image_size))
            full_images = torch.from_numpy(full_images)[:, None, :, :].expand((-1, 3, -1, -1))
            state = {
                'images': full_images,
                'num_frames': all_datasets_num,
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
            self.masks = self.masks.to('cuda')
            self.mask_points = self.train_dataset._dataparser_outputs.metadata.get('mask_points', {})
            with torch.no_grad():
                sam2predictor._get_image_feature(state, frame_idx=0, batch_size=1)
                for k, v in {'plant': 192, 'hemisphere': 127, 'hemisphere2': 128, 'board': 64}.items():
                    if self.mask_points.get(k, None) is not None:
                        sam2predictor.add_new_points_or_box(state, frame_idx=0, obj_id=v,
                                                            points=torch.tensor(self.mask_points[k]),
                                                            labels=torch.tensor([1, ] * len(self.mask_points[k])))
                for out_frame_idx, out_obj_ids, out_mask_logits in sam2predictor.propagate_in_video(state):
                    for obj_id, out_obj_id in enumerate(out_obj_ids):
                        mask = out_mask_logits[obj_id, 0] > 0.0
                        self.masks[out_frame_idx][mask] = out_obj_id
                self.masks = self.masks.to('cpu')
                mask_save_path = self.config.data.parent / 'masks'
                if not os.path.exists(mask_save_path):
                    os.makedirs(mask_save_path)
                for i in range(all_datasets_num):
                    cv2.imwrite(str(mask_save_path / f'{i:04d}.png'), self.masks[i, ...].numpy())
                self.masks = self.masks[np.concatenate((i_train, i_eval), axis=0)]
            return
        for i, mask in enumerate(mask_filenames):
            self.masks[i, ...] = torch.from_numpy(cv2.imread(str(mask), cv2.IMREAD_UNCHANGED))
        self.masks = self.masks[np.concatenate((i_train, i_eval), axis=0)]

    def next_train(self, step: int) -> Tuple[RayBundle, Dict]:
        if step % 2 == 0:
            indices = self.sample_index(self.config.train_num_rays_per_batch, (360, 1085, 2045))
        else:
            indices = self.sample_index(self.config.train_num_rays_per_batch, (360, 1085, 2045),
                                        self.masks[: self.train_data.shape[0]], 255)
        ray_bundle = self.train_ray_generator(indices)

        c, y, x = (i.flatten() for i in torch.split(indices, 1, dim=-1))
        dy, dx = y // 5, x // 5
        dy, dx = (dy[:, None, None] * 5 + torch.arange(0, 5)[None, :, None],
                  dx[:, None, None] * 5 + torch.arange(0, 5)[None, None, :])
        batch = {
            'image': self.train_data[c, y, x][:, None],
            'indices': indices,
            'sam': self.train_data[c[:, None, None], dy, dx].reshape(-1, 25),
        }
        self.train_count += 1
        return ray_bundle, batch

    def next_eval(self, step: int) -> Tuple[RayBundle, Dict]:
        indices = self.sample_index(self.config.eval_num_rays_per_batch, (40, 1085, 2045),
                                    self.masks[-self.eval_data.shape[0]:], 255
                                    )
        ray_bundle = self.eval_ray_generator(indices)
        c, y, x = (i.flatten() for i in torch.split(indices, 1, dim=-1))

        dy, dx = y // 5, x // 5
        dy, dx = (dy[:, None, None] * 5 + torch.arange(0, 5)[None, :, None],
                  dx[:, None, None] * 5 + torch.arange(0, 5)[None, None, :])

        batch = {
            'image': self.eval_data[c, y, x][:, None],
            'indices': indices,
            'sam': self.eval_data[c[:, None, None], dy, dx].reshape(-1, 25),
        }
        self.eval_count += 1
        return ray_bundle, batch

    def next_inference(self, mask_index=None, split='train') -> Tuple[RayBundle, Dict]:
        if split == 'train':
            shape = (360, 1085, 2045)
            mask = self.masks[:self.train_data.shape[0]]
            ray_generator = self.train_ray_generator
        elif split == 'val':
            shape = (40, 1085, 2045)
            mask = self.masks[-self.eval_data.shape[0]:]
            ray_generator = self.eval_ray_generator
        else:
            raise ValueError
        if mask_index is None:
            indices = self.sample_index(self.config.eval_num_rays_per_batch, shape)
        else:
            indices = self.sample_index(self.config.eval_num_rays_per_batch, shape, mask, mask_index)
        ray_bundle = ray_generator(indices)
        batch = {
            'indices': indices,
        }
        return ray_bundle, batch

    def sample_index(self, batch_size, shape, masks=None, mask_index=None):
        if masks is None:
            indices = (torch.rand(batch_size, 3) * torch.tensor(shape)).long()
            return indices
        # assert list(shape) == list(masks.shape)
        sampled = 0
        indices = torch.empty((batch_size, 3), dtype=torch.int64)
        while sampled < batch_size:
            indices_try = (torch.rand(batch_size, 3) * torch.tensor(shape)).long()
            c, y, x = (i.flatten() for i in torch.split(indices_try, 1, dim=-1))
            if mask_index == 255:
                try_indices_validity = masks[c, y, x] != 0
            else:
                try_indices_validity = masks[c, y, x] == mask_index
            validity_num = torch.sum(try_indices_validity)
            if validity_num + sampled <= batch_size:
                indices[sampled:sampled + validity_num, :] = indices_try[try_indices_validity, :]
            else:
                indices[-validity_num:, :] = indices_try[try_indices_validity, :]
            sampled += validity_num
        return indices

    def next_eval_image(self, step: int) -> Tuple[Cameras, Dict]:
        raise NotImplementedError

    def get_train_rays_per_batch(self) -> int:
        return self.config.train_num_rays_per_batch

    def get_eval_rays_per_batch(self) -> int:
        return self.config.eval_num_rays_per_batch

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        return {}
