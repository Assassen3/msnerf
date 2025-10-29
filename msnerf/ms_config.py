from nerfstudio.data.datamanagers.parallel_datamanager import ParallelDataManagerConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig
from nerfstudio.plugins.types import MethodSpecification

from msnerf.ms_datamanager import MSParallelDataManager
from msnerf.ms_dataparser import MSDataParserConfig
from msnerf.ms_model import MSNerfModelConfig

MsNeRFMethod = MethodSpecification(
    config=TrainerConfig(
        method_name="msnerf",
        steps_per_save=10000,
        max_num_iterations=10000,
        steps_per_eval_batch=500,
        steps_per_eval_image=1000,
        mixed_precision=True,
        pipeline=VanillaPipelineConfig(
            datamanager=ParallelDataManagerConfig(
                _target=MSParallelDataManager,
                dataparser=MSDataParserConfig(
                    keep_coord=False
                ),
                train_num_rays_per_batch=1 << 13,
                eval_num_rays_per_batch=1 << 14,
            ),
            model=MSNerfModelConfig(
                num_multispectral=25,
                eval_num_rays_per_chunk=1 << 15,
                num_nerf_samples_per_ray=32,
                num_proposal_samples_per_ray=(256, 128),
                hidden_dim=128,
                hidden_dim_ms=128,
                average_init_density=0.01,
                log2_hashmap_size=18
            ),
        ),
        optimizers={
            "proposal_networks": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=200000),
            },
            "fields": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=200000),
            },
            "camera_opt": {
                "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-4, max_steps=5000),
            },
        },
        vis="tensorboard",
    ),
    description="Multispectral NeRF"
)
