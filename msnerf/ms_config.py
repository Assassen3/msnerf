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
        steps_per_save=2000,
        max_num_iterations=2000,
        steps_per_eval_batch=3000,
        steps_per_eval_image=3000,
        mixed_precision=True,
        pipeline=VanillaPipelineConfig(
            datamanager=ParallelDataManagerConfig(
                _target=MSParallelDataManager,
                dataparser=MSDataParserConfig(),
                train_num_rays_per_batch=1 << 14,
                eval_num_rays_per_batch=1 << 15,
            ),
            model=MSNerfModelConfig(senmantic=True),
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
