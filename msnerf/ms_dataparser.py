from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Type

import numpy as np
import torch
from nerfstudio.cameras import camera_utils
from nerfstudio.cameras.cameras import CameraType, Cameras
from nerfstudio.data.dataparsers.base_dataparser import DataParser, DataParserConfig, DataparserOutputs
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.data.utils.dataparsers_utils import (
    get_train_eval_split_fraction,
)
from nerfstudio.utils.io import load_from_json


@dataclass
class MSDataParserConfig(DataParserConfig):
    _target: Type = field(default_factory=lambda: MSDatasetParser)
    train_split_fraction: float = 0.9
    scene_scale: float = 2.0


@dataclass
class MSDatasetParser(DataParser):
    config: MSDataParserConfig

    def _generate_dataparser_outputs(self, split="train", **kwargs: Optional[Dict]):
        assert self.config.data.exists(), f"Data directory {self.config.data} does not exist."

        if self.config.data.suffix == ".json":
            meta = load_from_json(self.config.data)
            data_dir = self.config.data.parent
        else:
            meta = load_from_json(self.config.data / "transforms.json")
            data_dir = self.config.data

        image_filenames = []
        mask_filenames = []
        poses = []

        # sort the frames by fname
        fnames = []
        for frame in meta["frames"]:
            filepath = Path(frame["file_path"])
            fname = data_dir / filepath
            fnames.append(fname)
        inds = np.argsort(fnames)
        frames = [meta["frames"][ind] for ind in inds]

        for frame in frames:
            filepath = Path(frame["file_path"])
            fname = data_dir / filepath

            image_filenames.append(fname)
            poses.append(np.array(frame["transform_matrix"]))
            if 'mask_path' in meta:
                mask_name = Path(meta["mask_path"]) / fname.name
                mask_filenames.append(mask_name)

        i_train, i_eval = get_train_eval_split_fraction(image_filenames, self.config.train_split_fraction)

        if split == "train":
            indices = i_train
        elif split in ["val", "test"]:
            indices = i_eval
        else:
            raise ValueError(f"Unknown dataparser split {split}")

        poses = torch.from_numpy(np.array(poses).astype(np.float32))

        image_filenames = [image_filenames[i] for i in indices]

        idx_tensor = torch.tensor(indices, dtype=torch.long)
        poses = poses[idx_tensor]

        aabb_scale = self.config.scene_scale
        scene_box = SceneBox(
            aabb=torch.tensor(
                [[-aabb_scale, -aabb_scale, -aabb_scale], [aabb_scale, aabb_scale, aabb_scale]], dtype=torch.float32
            )
        )

        fx = float(meta["fl_x"])
        fy = float(meta["fl_y"])
        cx = float(meta["cx"])
        cy = float(meta["cy"])
        height = int(meta["h"])
        width = int(meta["w"])
        num_channels = int(meta["num_ms"])
        distortion_params = (camera_utils.get_distortion_params())
        camera_type = CameraType.PERSPECTIVE
        metadata = {"num_ms": num_channels}
        mask_points = meta.get("mask_points", None)

        cameras = Cameras(
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            distortion_params=distortion_params,
            height=height,
            width=width,
            camera_to_worlds=poses[:, :3, :4],
            camera_type=camera_type,
            metadata=metadata,
        )

        dataparser_outputs = DataparserOutputs(
            image_filenames=image_filenames,
            cameras=cameras,
            scene_box=scene_box,
            mask_filenames=mask_filenames if len(mask_filenames) > 0 else None,
            metadata={'mask_points': mask_points},
        )
        return dataparser_outputs
