from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig
from nerfstudio.plugins.types import MethodSpecification

from msnerf.ms_datamanager import MaskedDataManager, MaskedDataManagerConfig
from msnerf.ms_dataparser import MSDataParserConfig
from msnerf.ms_model import MSNerfModelConfig

MsNeRFMethod = MethodSpecification(
    config=TrainerConfig(
        method_name="msnerf",
        steps_per_save=3000,
        max_num_iterations=3000,
        steps_per_eval_batch=0,
        steps_per_eval_image=0,
        steps_per_eval_all_images=0,
        mixed_precision=True,
        use_grad_scaler=True,
        pipeline=VanillaPipelineConfig(
            datamanager=MaskedDataManagerConfig(
                _target=MaskedDataManager,
                dataparser=MSDataParserConfig(),
                sam2_ckpt_path=r"D:\files\PHD\myNeRF\Grounded-SAM-2\checkpoints\sam2.1_hiera_small.pt",
            ),
            model=MSNerfModelConfig(senmantic=True),
        ),
        optimizers={
            "proposal_networks": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=2000),
            },
            "fields": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=2000),
            },
            "camera_opt": {
                "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-4, max_steps=500),
            },
        },
        vis='tensorboard',
    ),
    description="Multispectral NeRF"
)
