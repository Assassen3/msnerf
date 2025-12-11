from typing import Optional, Tuple, Union

import nerfacc
import torch
from jaxtyping import Float, Int
from torch import Tensor, nn


class MSRenderer(nn.Module):
    def __init__(self, num_ms: int, background_color="random", semantic=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_ms = num_ms
        self.background_color = background_color
        self.semantic = semantic

    def combine_ms(
            self,
            ms: Float[Tensor, "*bs num_samples num_ms"],
            weights: Float[Tensor, "*bs num_samples 1"],
            background_color="random",
            ray_indices: Optional[Int[Tensor, "num_samples"]] = None,
            num_rays: Optional[int] = None,
    ) -> Float[Tensor, "*bs num_ms"]:
        if ray_indices is not None and num_rays is not None:
            # Necessary for packed samples from volumetric ray sampler
            if background_color == "last_sample":
                raise NotImplementedError("Background color 'last_sample' not implemented for packed samples.")
            comp_ms = nerfacc.accumulate_along_rays(
                weights[..., 0], values=ms, ray_indices=ray_indices, n_rays=num_rays
            )
            accumulated_weight = nerfacc.accumulate_along_rays(
                weights[..., 0], values=None, ray_indices=ray_indices, n_rays=num_rays
            )
        else:
            comp_ms = torch.sum(weights * ms, dim=-2)
            accumulated_weight = torch.sum(weights, dim=-2)
        if background_color == "random":
            return comp_ms
        elif background_color == "last_sample":
            # Note, this is only supported for non-packed samples.
            background_color = ms[..., -1, :]
        background_color = self.get_background_color(background_color, shape=comp_ms.shape, device=comp_ms.device)

        assert isinstance(background_color, torch.Tensor)
        comp_ms = comp_ms + background_color * (1.0 - accumulated_weight)
        return comp_ms

    def get_background_color(
            self, background_color, shape: Tuple[int, ...], device: torch.device
    ) -> Union[Float[Tensor, "3"], Float[Tensor, "*bs 3"]]:
        assert background_color not in {"last_sample", "random"}
        # assert shape[-1] == 3, "Background color must be RGB."
        if isinstance(background_color, str) and background_color == "ms_black":
            background_color = torch.tensor([0.0, ], device=device)
        assert isinstance(background_color, Tensor)

        return background_color.expand(shape).to(device)

    def blend_background(
            self,
            image: Tensor,
            background_color=None,
    ) -> Float[Tensor, "*bs num_ms"]:
        '''
        对真值数图片背景处理
        不开semantic就返回原图
        开了semantic就会对背景部分涂黑(metrics)，或随机颜色(loss)
        :param image:
        :param background_color:
        :return:
        '''
        if not self.semantic:
            return image
        opacity = (image > 0).to(image)
        assert self.background_color == "random"
        if background_color is None:
            background_color = self.background_color
            if background_color in {"last_sample", "random"}:
                background_color = "ms_black"
        background_color = self.get_background_color(background_color, shape=image.shape, device=image.device)
        assert isinstance(background_color, torch.Tensor)
        return image * opacity + background_color.to(image.device) * (1 - opacity)

    def blend_background_for_loss_computation(
            self,
            pred_image: Tensor,
            pred_accumulation: Tensor,
            ms_index: Tensor,
            gt_image: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        '''
        在训练时使用，将背景透明化
        :param pred_image:
        :param pred_accumulation:
        :param ms_index:
        :param gt_image:
        :return:
        '''
        # pred_image = self._extract_band(pred_image, ms_index)
        background_color = self.background_color
        if background_color == "last_sample":
            raise NotImplementedError  # 只做测试，可删
            background_color = "black"  # No background blending for GT
        elif background_color == "random":
            background_color = torch.rand_like(pred_image)
            pred_image = pred_image + background_color * (1.0 - pred_accumulation)
        gt_image = self.blend_background(gt_image, background_color=background_color)
        return pred_image, gt_image

    def _extract_band(self, pred_image: Tensor, ms_index: Tensor) -> Tensor:
        assert pred_image.shape[:-1] == ms_index.shape[:-1]
        return torch.gather(pred_image, -1, ms_index)

    def forward(
            self,
            ms: Float[Tensor, "*bs num_samples num_ms"],
            weights: Float[Tensor, "*bs num_samples 1"],
            ray_indices: Optional[Int[Tensor, "num_samples"]] = None,
            num_rays: Optional[int] = None,
            background_color=None,
    ) -> Float[Tensor, "*bs num_ms"]:
        if background_color is None:
            background_color = self.background_color

        if not self.training:
            ms = torch.nan_to_num(ms)
        ms = self.combine_ms(
            ms, weights, background_color=background_color, ray_indices=ray_indices, num_rays=num_rays
        )
        if not self.training:
            torch.clamp_(ms, min=0.0, max=1.0)
        return ms
