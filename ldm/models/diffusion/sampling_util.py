import torch
import numpy as np


def append_dims(x: torch.Tensor, target_dims: int) -> torch.Tensor:
    """Appends dimensions to the end of a tensor until it has target_dims dimensions.
    From https://github.com/crowsonkb/k-diffusion/blob/master/k_diffusion/utils.py"""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(f'Input tensor has {x.ndim} dimensions but target_dims is {target_dims}, which is less than the number of dimensions in the input tensor.')
    return x[(...,) + (None,) * dims_to_append]


def norm_thresholding(x0: torch.Tensor, value: float) -> torch.Tensor:
    s = append_dims(x0.pow(2).flatten(1).mean(1).sqrt().clamp(min=value), x0.ndim)
    return x0 * (value / s)


def spatial_norm_thresholding(x0: torch.Tensor, value: float) -> torch.Tensor:
    # b c h w
    s = x0.pow(2).mean(1, keepdim=True).sqrt().clamp(min=value)
    return x0 * (value / s)

