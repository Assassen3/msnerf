from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Type
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.cameras.rays import RayBundle, RaySamples
from nerfstudio.data.scene_box import OrientedBox
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes, TrainingCallbackLocation
from nerfstudio.field_components.field_heads import (
    FieldHeadNames, )
from nerfstudio.field_components.spatial_distortions import SceneContraction
from nerfstudio.fields.density_fields import HashMLPDensityField
from nerfstudio.model_components.losses import distortion_loss, interlevel_loss
from nerfstudio.model_components.ray_samplers import ProposalNetworkSampler
from nerfstudio.model_components.renderers import AccumulationRenderer, DepthRenderer, NormalsRenderer
from nerfstudio.model_components.scene_colliders import NearFarCollider
from nerfstudio.models.base_model import Model, ModelConfig
from torch.nn import MSELoss
from torch.nn import Parameter
from torchmetrics.functional import mean_squared_error, peak_signal_noise_ratio, structural_similarity_index_measure
from torchmetrics.image import LearnedPerceptualImagePatchSimilarity

from msnerf.ms_field import MSNerfField
from msnerf.ms_renderer import MSRenderer
from msnerf.tools.ms_export import generate_ms_point_cloud


@dataclass
class MSNerfModelConfig(ModelConfig):
    _target: Type = field(default_factory=lambda: MSNerfModel)
    num_multispectral: int = 25
    num_levels: int = 16
    base_res: int = 16
    max_res: int = 2048
    log2_hashmap_size: int = 19
    features_per_level: int = 2
    num_layers: int = 2
    hidden_dim: int = 128
    geo_feat_dim: int = 31
    num_layers_ms: int = 3
    hidden_dim_ms: int = 128
    implementation: Literal["tcnn", "torch"] = "tcnn"
    camera_optimizer: CameraOptimizerConfig = field(default_factory=lambda: CameraOptimizerConfig(mode="SO3xR3"))
    use_proposal_weight_anneal: bool = True
    proposal_net_args_list: List[Dict] = field(
        default_factory=lambda: [
            {"hidden_dim": 16, "log2_hashmap_size": 17, "num_levels": 5, "max_res": 128, "use_linear": False},
            {"hidden_dim": 16, "log2_hashmap_size": 17, "num_levels": 5, "max_res": 256, "use_linear": False},
        ]
    )
    average_init_density: float = 0.01

    proposal_update_every: int = 5
    proposal_warmup: int = 5000
    num_nerf_samples_per_ray: int = 48
    num_proposal_samples_per_ray: Tuple[int, ...] = (256, 128)
    use_single_jitter: bool = True
    background_color: Literal["random", "last_sample", "black", "white", "ms_white", "ms_black"] = "random"
    proposal_weights_anneal_max_num_iters: int = 1000
    proposal_weights_anneal_slope: float = 10.0

    interlevel_loss_mult: float = 1.0
    distortion_loss_mult: float = 0.002
    orientation_loss_mult: float = 0.0001
    pred_normal_loss_mult: float = 0.001
    eval_num_rays_per_chunk: int = 1 << 16

    senmantic: bool = False


class MSNerfModel(Model):
    config: MSNerfModelConfig

    def populate_modules(self):
        super().populate_modules()

        scene_contraction = SceneContraction(order=float("inf"))
        self.field = MSNerfField(
            aabb=self.scene_box.aabb,
            num_multispectral=self.config.num_multispectral,
            num_levels=self.config.num_levels,
            base_res=self.config.base_res,
            max_res=self.config.max_res,
            log2_hashmap_size=self.config.log2_hashmap_size,
            features_per_level=self.config.features_per_level,
            num_layers=self.config.num_layers,
            hidden_dim=self.config.hidden_dim,
            geo_feat_dim=self.config.geo_feat_dim,
            hidden_dim_ms=self.config.hidden_dim_ms,
            num_layers_ms=self.config.num_layers_ms,
            implementation=self.config.implementation,
            spatial_distortion=scene_contraction,
        )

        # self.camera_optimizer: CameraOptimizer = self.config.camera_optimizer.setup(
        #     num_cameras=self.num_train_data, device="cpu"
        # )

        self.density_fns = []
        self.proposal_networks = torch.nn.ModuleList()
        for i in range(len(self.config.proposal_net_args_list)):
            prop_net_args = self.config.proposal_net_args_list[i]
            network = HashMLPDensityField(
                self.scene_box.aabb,
                spatial_distortion=scene_contraction,
                **prop_net_args,
                average_init_density=self.config.average_init_density,
                implementation=self.config.implementation,
            )
            self.proposal_networks.append(network)
        self.density_fns.extend([network.density_fn for network in self.proposal_networks])

        def update_schedule(step):
            return np.clip(
                np.interp(step, [0, self.config.proposal_warmup], [0, self.config.proposal_update_every]),
                1,
                self.config.proposal_update_every,
            )

        self.proposal_sampler = ProposalNetworkSampler(
            num_nerf_samples_per_ray=self.config.num_nerf_samples_per_ray,
            num_proposal_samples_per_ray=self.config.num_proposal_samples_per_ray,
            num_proposal_network_iterations=len(self.config.proposal_net_args_list),
            single_jitter=self.config.use_single_jitter,
            update_sched=update_schedule,
            initial_sampler=None,  # important!
        )

        self.collider = NearFarCollider(near_plane=0.01, far_plane=100)

        self.renderer_ms = MSRenderer(background_color=self.config.background_color,
                                      num_ms=self.config.num_multispectral,
                                      semantic=self.config.senmantic)
        self.renderer_accumulation = AccumulationRenderer()

        self.renderer_depth = DepthRenderer(method="median")
        # self.renderer_expected_depth = DepthRenderer(method="expected")
        self.renderer_normals = NormalsRenderer()

        self.ms_loss = MSELoss()
        self.step = 0

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = {}
        param_groups["proposal_networks"] = list(self.proposal_networks.parameters())
        param_groups["fields"] = list(self.field.parameters())
        return param_groups

    def get_training_callbacks(
            self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        callbacks = []
        # anneal the weights of the proposal network before doing PDF sampling
        if self.config.use_proposal_weight_anneal:
            # anneal the weights of the proposal network before doing PDF sampling
            N = self.config.proposal_weights_anneal_max_num_iters

            def set_anneal(step):
                # https://arxiv.org/pdf/2111.12077.pdf eq. 18
                self.step = step
                train_frac = np.clip(step / N, 0, 1)
                self.step = step

                def bias(x, b):
                    return b * x / ((b - 1) * x + 1)

                anneal = bias(train_frac, self.config.proposal_weights_anneal_slope)
                self.proposal_sampler.set_anneal(anneal)

            callbacks.append(
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                    update_every_num_iters=1,
                    func=set_anneal,
                )
            )
            callbacks.append(
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                    update_every_num_iters=1,
                    func=self.proposal_sampler.step_cb,
                )
            )

            def export_pointcloud(step: int):
                generate_ms_point_cloud(pipeline=training_callback_attributes.pipeline,
                                        output_dir=training_callback_attributes.trainer.config.get_base_dir(),
                                        num_points=200000)

            callbacks.append(
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.AFTER_TRAIN],
                    func=export_pointcloud
                )
            )
            # def cali(step: int):

        return callbacks

    def get_outputs(self, ray_bundle: RayBundle):
        ray_samples: RaySamples
        ray_samples, weights_list, ray_samples_list = self.proposal_sampler(ray_bundle, density_fns=self.density_fns)
        field_outputs = self.field.forward(ray_samples)

        weights = ray_samples.get_weights(field_outputs[FieldHeadNames.DENSITY])
        weights_list.append(weights)
        ray_samples_list.append(ray_samples)

        ms = self.renderer_ms(ms=field_outputs["ms"], weights=weights)

        with torch.no_grad():
            depth = self.renderer_depth(weights=weights, ray_samples=ray_samples)
        # expected_depth = self.renderer_expected_depth(weights=weights, ray_samples=ray_samples)
        accumulation = self.renderer_accumulation(weights=weights)

        outputs = {
            "ms": ms,
            "accumulation": accumulation,
            "depth": depth,
            # "expected_depth": expected_depth,
        }

        if ray_bundle.metadata.get("ms_index", None) is not None:
            outputs["ms_index"] = ray_bundle.metadata["ms_index"]

        if self.training:
            outputs["weights_list"] = weights_list
            outputs["ray_samples_list"] = ray_samples_list

        for i in range(len(self.config.proposal_net_args_list)):
            outputs[f"prop_depth_{i}"] = self.renderer_depth(weights=weights_list[i], ray_samples=ray_samples_list[i])
        return outputs

    def get_normals(self, ray_bundle: RayBundle, off=0.001):
        with torch.no_grad():
            offsets = torch.tensor([
                [-off, -off, 0], [-off, 0, 0], [-off, off, 0],
                [0, -off, 0], [0, 0, 0], [0, off, 0],
                [off, -off, 0], [off, 0, 0], [off, off, 0]
            ], dtype=torch.float32, device=self.device)[None, :, :]
            origins = ray_bundle.origins[:, None, :].expand(-1, 9, -1)
            directions = ray_bundle.directions[:, None, :].expand(-1, 9, -1)
            origins = (origins + torch.cross(directions, offsets, dim=-1))
            metadata = {}
            for k, v in ray_bundle.metadata.items():
                if isinstance(v, torch.Tensor):
                    metadata[k] = v[:, None, :]
                else:
                    metadata[k] = v
            ray_bundle = RayBundle(
                origins=origins,
                directions=directions,
                pixel_area=ray_bundle.pixel_area[:, None, :],
                camera_indices=ray_bundle.camera_indices[:, None, :],
                metadata=metadata,
            )
            depths = torch.empty(list(ray_bundle.shape) + [1], dtype=torch.float32, device=self.device)
            for i in range(9):
                depths[:, i, :] = self(ray_bundle[:, i])['depth']
            point = origins + depths * directions
            p_mean = point.mean(dim=1, keepdim=True)
            p_centered = point - p_mean
            C = torch.matmul(p_centered.transpose(1, 2), p_centered)
            C = C.contiguous()
            eigenvalues, eigenvectors = torch.linalg.eigh(C)
            normal = eigenvectors[:, :, 0]  # [B, 3]
            normal = F.normalize(normal, dim=-1)  # [B, 3]
            center_dirs = directions.reshape(-1, 9, 3)[:, 4, :]
            dot = (normal * center_dirs).sum(dim=-1, keepdim=True)  # [B, 1]
            flip_mask = dot > 0
            normal = torch.where(flip_mask, -normal, normal)
            return normal

    def get_metrics_dict(self, outputs, batch):
        metrics_dict = {}
        # gt_ms = batch["image"].to(self.device)
        # gt_ms = self.renderer_ms.blend_background(gt_ms)  # RGB or RGBA image
        # predicted_ms = outputs["ms"]
        # metrics_dict["psnr"] = self.psnr(predicted_ms, gt_ms)
        if 'sam' in batch:
            cos = F.cosine_similarity(outputs['ms'], batch['sam'].to('cuda'), dim=-1)
            cos = torch.acos(torch.mean(cos)) * 180 / torch.pi
            metrics_dict['sam'] = cos
        if self.training:
            metrics_dict["distortion"] = distortion_loss(outputs["weights_list"], outputs["ray_samples_list"])

        return metrics_dict

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        loss_dict = {}
        image = batch["sam"].to(self.device)
        pred_ms, gt_ms = self.renderer_ms.blend_background_for_loss_computation(
            pred_image=outputs["ms"],
            ms_index=outputs["ms_index"],
            pred_accumulation=outputs["accumulation"],
            gt_image=image,
        )

        loss_dict["ms_loss"] = self.ms_loss(gt_ms, pred_ms)
        # loss_dict["sam_loss"] = sam_loss(pred_ms, gt_ms) * self.config.sam_loss_mult
        if self.training:
            loss_dict["interlevel_loss"] = self.config.interlevel_loss_mult * interlevel_loss(
                outputs["weights_list"], outputs["ray_samples_list"]
            )
            assert metrics_dict is not None and "distortion" in metrics_dict
            loss_dict["distortion_loss"] = self.config.distortion_loss_mult * metrics_dict["distortion"]
        return loss_dict

    def get_image_metrics_and_images(
            self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        gt_ms = batch["image"].to(self.device)
        predicted_ms = self.renderer_ms._extract_band(outputs["ms"], outputs["ms_index"])

        band_per_row = int(self.config.num_multispectral ** 0.5)
        metrics_dict = {}
        for band in range(self.config.num_multispectral):
            gt_band = gt_ms[band // band_per_row:: band_per_row, band % band_per_row:: band_per_row][
                None, None, ...].contiguous()
            pred_band = predicted_ms[band // band_per_row:: band_per_row, band % band_per_row:: band_per_row].squeeze()[
                None, None, ...].contiguous()

            ssim = structural_similarity_index_measure(pred_band, gt_band)
            psnr = peak_signal_noise_ratio(pred_band, gt_band, data_range=1.0)
            mse = mean_squared_error(pred_band, gt_band)
            lpips_class = LearnedPerceptualImagePatchSimilarity().to('cuda')
            lpips = lpips_class(pred_band.repeat([1, 3, 1, 1]) * 2 - 1, gt_band.repeat([1, 3, 1, 1]) * 2 - 1)
            # sam = spectral_angle_mapper(pred_band, gt_band)

            metrics_dict.update({f"psnr": float(psnr.item()),
                                 f"ssim": float(ssim.item()),
                                 f"mse": float(mse.item()),
                                 f"lpips": float(lpips.item())
                                 # f"{band}_sam": float(sam.item())
                                 })
            break

        return metrics_dict, {}

    def get_rgba_image(self, outputs: Dict[str, torch.Tensor], output_name: str = "ms") -> torch.Tensor:
        ms = outputs[output_name]
        acc = outputs["accumulation"]
        if acc.dim() < ms.dim():
            acc = acc.unsqueeze(-1)
        return torch.cat((ms, acc), dim=-1)

    def get_outputs_for_camera(self, camera: Cameras, obb_box: Optional[OrientedBox] = None) -> Dict[str, torch.Tensor]:
        with torch.no_grad():
            ray_bundle = camera.generate_rays(camera_indices=0, keep_shape=True, obb_box=obb_box)
            ms_per_row = int(camera.metadata['num_ms'] ** 0.5)
            ms_index = (torch.arange(ray_bundle.shape[1], device=camera.device)[None, :] % ms_per_row + (
                    torch.arange(ray_bundle.shape[0], device=camera.device)[:, None] % ms_per_row) * ms_per_row)
            ray_bundle.metadata['ms_index'] = ms_index[..., None]
            num_rays_per_chunk = self.config.eval_num_rays_per_chunk
            image_height, image_width = ray_bundle.origins.shape[:2]
            num_rays = len(ray_bundle)
            outputs_lists = defaultdict(list)
            for i in range(0, num_rays, num_rays_per_chunk):
                start_idx = i
                end_idx = i + num_rays_per_chunk
                ray_bundle_chunk = ray_bundle.get_row_major_sliced_ray_bundle(start_idx, end_idx)
                ray_bundle_chunk = ray_bundle_chunk.to(self.device)
                outputs = self.forward(ray_bundle=ray_bundle_chunk)
                for output_name, output in outputs.items():  # type: ignore
                    outputs_lists[output_name].append(output)
            outputs = {}
            for output_name, outputs_list in outputs_lists.items():
                outputs[output_name] = torch.cat(outputs_list).view(image_height, image_width, -1)  # type: ignore
            return outputs
    def get_outputs_for_reduced_camera(self, camera: Cameras, obb_box: Optional[OrientedBox] = None) -> Dict[str, torch.Tensor]:
        with torch.no_grad():
            ray_bundle = camera.generate_rays(camera_indices=0, keep_shape=True, obb_box=obb_box)
            reduced_ray_bundle = ray_bundle[2::5, 2::5]
            # ray_bundle.metadata['ms_index'] = ms_index[..., None]
            num_rays_per_chunk = self.config.eval_num_rays_per_chunk
            image_height, image_width = reduced_ray_bundle.origins.shape[:2]
            num_rays = len(reduced_ray_bundle)
            outputs_lists = defaultdict(list)
            for i in range(0, num_rays, num_rays_per_chunk):
                start_idx = i
                end_idx = i + num_rays_per_chunk
                ray_bundle_chunk = reduced_ray_bundle.get_row_major_sliced_ray_bundle(start_idx, end_idx)
                ray_bundle_chunk = ray_bundle_chunk.to(self.device)
                outputs = self.forward(ray_bundle=ray_bundle_chunk)
                for output_name, output in outputs.items():  # type: ignore
                    outputs_lists[output_name].append(output)
            outputs = {}
            for output_name, outputs_list in outputs_lists.items():
                outputs[output_name] = torch.cat(outputs_list).view(image_height, image_width, -1)  # type: ignore
            return outputs