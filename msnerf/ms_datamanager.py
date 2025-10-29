from __future__ import annotations

import torch
from nerfstudio.data.datamanagers.parallel_datamanager import ParallelDataManager, DataProcessor
from nerfstudio.model_components.ray_generators import RayGenerator
from pathos.helpers import mp

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


class MSDataProcessor(DataProcessor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.ray_generator = MSRayGenerator(self.dataset.cameras)


class MSParallelDataManager(ParallelDataManager[MSSRDataset]):
    def setup_train(self):
        assert self.train_dataset is not None
        self.train_pixel_sampler = self._get_pixel_sampler(self.train_dataset, self.config.train_num_rays_per_batch)
        self.data_queue = mp.Queue(maxsize=self.config.queue_size)
        self.data_procs = [
            MSDataProcessor(
                out_queue=self.data_queue,  # type: ignore
                config=self.config,
                dataparser_outputs=self.train_dataparser_outputs,
                dataset=self.train_dataset,
                pixel_sampler=self.train_pixel_sampler,
            )
            for i in range(self.config.num_processes)
        ]
        for proc in self.data_procs:
            proc.start()
        print("Started threads")

    def setup_eval(self):
        super().setup_eval()
        self.eval_ray_generator = MSRayGenerator(self.eval_dataset.cameras.to(self.device))
