"""PyTorch implementation of equations (1)-(5) in the supplied screenshot.

The screenshot, not a generic NAMM implementation, defines the equations here.
Supplementary 5x5 filters: https://pmc.ncbi.nlm.nih.gov/articles/PMC8320967/
See README_JND.md for unresolved choices and units.
"""

from __future__ import annotations

from collections.abc import Sequence
import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F
import numpy as np
import torchvision.io as io
import matplotlib.pyplot as plt


class CalculateJND(nn.Module):
    """Compute one JND threshold per input pixel and channel.

    Args:
        c_lt: Explicit overlap coefficient, scalar or one value per channel.
        value_range: 1 for float images in [0,1], 255 for images in [0,255].
            Thresholds are returned in these same units. Do not pass
            standardized (mean/std normalized) model inputs.
        pool_size: Odd, stride-one spatial max pooling size. The screenshot
            does not specify this size. 1 disables pooling.
        background: "weighted" uses the published 5x5 background filter;
            "mean" uses a uniform 5x5 average. Both operate channelwise.
        gradient_mode: "signed" implements Eq. (4) literally;
            "magnitude" takes abs before the directional maximum.
        padding_mode: Boundary convention, absent from the screenshot.

    Input: floating [N,C,H,W] or [C,H,W]. Channels are treated independently.
    Output: same shape; float64 stays float64, other floats use float32.
    The module contains no trainable parameters and supports autograd.
    """

    def __init__(
        self,
        *,
        c_lt: float | Sequence[float],
        value_range: float = 1.0,
        pool_size: int = 3,
        background: str = "weighted",
        gradient_mode: str = "signed",
        padding_mode: str = "replicate",
    ) -> None:
        super().__init__()
        if value_range not in (1, 255):
            raise ValueError("value_range must be 1 or 255")
        if isinstance(pool_size, bool) or not isinstance(pool_size, int):
            raise ValueError("pool_size must be a positive odd integer")
        if pool_size < 1 or pool_size % 2 == 0:
            raise ValueError("pool_size must be a positive odd integer")
        if background not in ("weighted", "mean"):
            raise ValueError("background must be 'weighted' or 'mean'")
        if gradient_mode not in ("signed", "magnitude"):
            raise ValueError("gradient_mode must be 'signed' or 'magnitude'")
        if padding_mode not in ("replicate", "reflect", "constant"):
            raise ValueError("unsupported padding_mode")
        coefficients = torch.as_tensor(c_lt, dtype=torch.float64).detach().clone()
        if coefficients.ndim > 1 or coefficients.numel() == 0:
            raise ValueError("c_lt must be a scalar or nonempty 1-D sequence")
        if not torch.isfinite(coefficients).all() or (coefficients < 0).any():
            raise ValueError("c_lt must be finite and nonnegative")
        self.register_buffer("c_lt", coefficients.reshape(-1))
        self.value_range = float(value_range)
        self.pool_size = pool_size
        self.background = background
        self.gradient_mode = gradient_mode
        self.padding_mode = padding_mode

        # Figure 2b: center excluded, inner neighbors doubled; sum = 32.
        bg = torch.tensor([
            [1, 1, 1, 1, 1],
            [1, 2, 2, 2, 1],
            [1, 2, 0, 2, 1],
            [1, 2, 2, 2, 1],
            [1, 1, 1, 1, 1],
        ], dtype=torch.float64) / 32
        if background == "mean":
            bg = torch.ones_like(bg) / 25
        self.register_buffer("background_kernel", bg[None, None])

        # Figure 3, G1-G4, with the 1/16 factor from screenshot Eq. (5).
        gradients = torch.tensor([
            [[0, 0, 0, 0, 0],
             [1, 3, 8, 3, 1],
             [0, 0, 0, 0, 0],
             [-1, -3, -8, -3, -1],
             [0, 0, 0, 0, 0]],
            [[0, 0, 1, 0, 0],
             [0, 8, 3, 0, 0],
             [1, 3, 0, -3, -1],
             [0, 0, -3, -8, 0],
             [0, 0, -1, 0, 0]],
            [[0, 0, 1, 0, 0],
             [0, 0, 3, 8, 0],
             [-1, -3, 0, 3, 1],
             [0, -8, -3, 0, 0],
             [0, 0, -1, 0, 0]],
            [[0, 1, 0, -1, 0],
             [0, 3, 0, -3, 0],
             [0, 8, 0, -8, 0],
             [0, 3, 0, -3, 0],
             [0, 1, 0, -1, 0]],
        ], dtype=torch.float64) / 16
        self.register_buffer("gradient_kernels", gradients[:, None])

    def _pad(self, x: Tensor, radius: int) -> Tensor:
        if self.padding_mode == "reflect":
            if min(x.shape[-2:]) <= radius:
                raise ValueError("reflect padding requires H and W > padding radius")
        return F.pad(x, (radius,) * 4, mode=self.padding_mode)

    def forward(
        self, x: Tensor, *, return_components: bool = False
    ) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
        if x.ndim not in (3, 4) or not x.is_floating_point():
            raise ValueError("expected a floating CHW or NCHW image tensor")
        if any(s == 0 for s in x.shape):
            raise ValueError("image dimensions must be nonempty")
        if not torch.isfinite(x).all():
            raise ValueError("image must contain only finite values")
        if (x < 0).any() or (x > self.value_range).any():
            print(x.min(), x.max())
            raise ValueError(f"input must lie in [0,{self.value_range:g}]")

        unbatched = x.ndim == 3
        if unbatched:
            x = x.unsqueeze(0)
        dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
        n, channels, h, w = x.shape
        if self.c_lt.numel() not in (1, channels):
            raise ValueError("c_lt must have one value or exactly C values")

        # All equations use 8-bit intensity units. Flatten N and C to
        # apply exactly the same filters independently to every channel.
        # Disable enclosing AMP so the threshold calculation stays >= fp32.
        with torch.autocast(device_type=x.device.type, enabled=False):
            image = x.to(dtype) * (255.0 / self.value_range)
            flat = image.reshape(n * channels, 1, h, w)
            padded = self._pad(flat, 2)
            bg_kernel = self.background_kernel.to(device=x.device, dtype=dtype)
            g_kernel = self.gradient_kernels.to(device=x.device, dtype=dtype)
            b = F.conv2d(padded, bg_kernel).reshape(n, channels, h, w)
            directional = F.conv2d(padded, g_kernel)
            if self.gradient_mode == "magnitude":
                directional = directional.abs()
            g = directional.amax(dim=1).reshape(n, channels, h, w)

            # Eq. (2), including its literal 21 and +3/128.
            # Finite derivative at black: preserve sqrt(0)=0 exactly,
            # but choose a zero subgradient at the singular endpoint.
            ratio = (b / 127.0).clamp_min(0)
            root = torch.where(
                ratio > 0,
                ratio.clamp_min(torch.finfo(dtype).tiny).sqrt(),
                torch.zeros_like(ratio),
            )
            t_l = torch.where(
                b <= 127.0,
                21.0 - 17.0 * root,
                (3.0 / 128.0) * b + 3.0 / 128.0,
            )
            # Eq. (3).
            t_t = (g / 100.0 - 1.0) * (b / 100.0 + 23.0 / 20.0) + 12.0
            c = self.c_lt.to(device=x.device, dtype=dtype).view(1, -1, 1, 1)
            # Eq. (1): deliberately NOT standard NAMM's min-overlap rule.
            raw = c * (t_t + 3.0) + t_l
            jnd_255 = raw
            if self.pool_size > 1:
                jnd_255 = F.max_pool2d(
                    self._pad(raw, self.pool_size // 2),
                    kernel_size=self.pool_size,
                    stride=1,
                )
            jnd = jnd_255 * (self.value_range / 255.0)

            # Normalize JND to [0, 1] range TODO check this works
            jnd_min = jnd.amin(dim=(-3, -2, -1), keepdim=True)
            jnd_max = jnd.amax(dim=(-3, -2, -1), keepdim=True)
            jnd = (jnd - jnd_min) / (jnd_max - jnd_min + 1e-8)

        def restore(t: Tensor) -> Tensor:
            return t.squeeze(0) if unbatched else t

        if not return_components:
            return restore(jnd)
        components = {
            "background_255": b,
            "max_gradient_255": g,
            "luminance_threshold_255": t_l,
            "texture_threshold_255": t_t,
            "raw_jnd_255": raw,
            "pooled_jnd_255": jnd_255,
        }
        
        return restore(jnd), {k: restore(v) for k, v in components.items()}


def project_to_jnd(
    candidate: Tensor,
    clean: Tensor,
    jnd: Tensor,
    *,
    eps: float,
    gamma: float = 1.0,
    value_range: float = 1.0,
    per_channel_max: bool = False,
) -> Tensor:
    """Project candidate onto the JND-calibrated L_inf ball around clean (Eqs. 8-10).

    Args:
        candidate: Unprojected adversarial tensor, shape [C, H, W] or [N, C, H, W].
        clean: Unnormalized clean image in [0, value_range], matching candidate.
        jnd: JND map computed from clean, matching candidate.
        eps: Base L_inf budget in the same units as clean (e.g. 8/255 for [0, 1]).
        gamma: Interpolation weight in [0, 1] between uniform eps (0) and rho (1).
        value_range: 1.0 for images in [0, 1], or 255.0 for images in [0, 255].
        per_channel_max: If True, compute JND_max per channel (H, W) instead of
            across the full image (C, H, W).
    """

    if not math.isfinite(eps) or eps < 0:
        raise ValueError("eps must be finite and nonnegative")
    if not math.isfinite(gamma) or not (0.0 <= gamma <= 1.0):
        raise ValueError("gamma must lie in [0, 1]")
    if value_range not in (1, 255):
        raise ValueError("value_range must be 1 or 255")
    if candidate.ndim not in (3, 4):
        raise ValueError("expected CHW or NCHW tensors")
    for t in (candidate, clean, jnd):
        if not t.is_floating_point() or not torch.isfinite(t).all():
            raise ValueError("inputs must be finite floating tensors")
        if t.shape != clean.shape or t.device != clean.device or t.dtype != clean.dtype:
            raise ValueError("candidate, clean, and jnd must match shape, device, and dtype")
    if (jnd < 0).any():
        raise ValueError("jnd thresholds must be nonnegative")
    if (clean < 0).any() or (clean > value_range).any():
        raise ValueError("clean pixels outside valid range")

    reduce_dims = (-2, -1) if per_channel_max else (-3, -2, -1)
    jnd_max = jnd.amax(dim=reduce_dims, keepdim=True).clamp_min(torch.finfo(jnd.dtype).tiny)
    # Eq. (8): rho = (JND / JND_max) * eps
    # Eqs. (9)-(10): eps_jnd = (1 - gamma) * eps + gamma * rho
    #                        = (1 + gamma * (JND / JND_max - 1)) * eps
    eps_jnd = (1.0 + gamma * (jnd / jnd_max - 1.0)) * eps
    lower = (clean - eps_jnd).clamp(min=0.0, max=value_range)
    upper = (clean + eps_jnd).clamp(min=0.0, max=value_range)
    return torch.clamp(candidate, min=lower, max=upper)


def visualize_jnd_matrix(image, thresholds):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 10))
    plt.subplot(2, 2, 1)
    print(image.shape, thresholds.shape)
    # Handle batch dimension
    if image.ndim == 4:
        image = image[0]
        thresholds = thresholds[0]
    plt.imshow(image.permute(1, 2, 0).cpu().numpy())
    plt.title('Original Image')
    plt.axis('off')
    # Calculate global min/max for consistent color scale
    vmin = thresholds.min().item()
    vmax = thresholds.max().item()
    plt.subplot(2, 2, 2)
    plt.imshow(thresholds[0, :, :].cpu().numpy(), cmap='viridis', vmin=vmin, vmax=vmax)
    plt.title('Red Channel (JND)')
    plt.colorbar(label='Intensity')
    plt.subplot(2, 2, 3)
    plt.imshow(thresholds[1, :, :].cpu().numpy(), cmap='viridis', vmin=vmin, vmax=vmax)
    plt.title('Green Channel (JND)')
    plt.colorbar(label='Intensity')
    plt.subplot(2, 2, 4)
    plt.imshow(thresholds[2, :, :].cpu().numpy(), cmap='viridis', vmin=vmin, vmax=vmax)
    plt.title('Blue Channel (JND)')
    plt.colorbar(label='Intensity')
    plt.tight_layout()
    plt.savefig('jnd_matrix.png')

def sample_patch_location(
    saliency,
    s,
    temperature=1.0,
    generator=None,
    device=None,
    return_prob=False,
):
    """
    Sample the top-left corner of an s x s patch.

    Probability is proportional to:
        mean saliency in the patch ** (1 / temperature)

    saliency: Tensor with shape (c, h, w), (1, h, w), or (h, w)
    """
    if saliency.ndim == 3:
        sal = saliency.mean(dim=0)
    elif saliency.ndim == 2:
        sal = saliency
    else:
        raise ValueError(
            "saliency must have shape (c, h, w), (1, h, w), or (h, w)."
        )

    if device is not None:
        sal = sal.to(device)

    sal = sal.to(torch.float64)
    h, w = sal.shape

    if not (1 <= s <= h and 1 <= s <= w):
        raise ValueError(
            f"Patch size {s} is incompatible with saliency map of shape {(h, w)}."
        )

    # Normalize to [0, 1] to avoid scale-dependent softmax behavior.
    sal_min = sal.min()
    sal_max = sal.max()

    if torch.isclose(sal_min, sal_max):
        # Uniform fallback if the saliency map is constant.
        window_probs = torch.full(
            (h - s + 1, w - s + 1),
            1.0 / ((h - s + 1) * (w - s + 1)),
            dtype=torch.float64,
            device=sal.device,
        )
    else:
        sal = (sal - sal_min) / (sal_max - sal_min)

        sal = sal.unsqueeze(0).unsqueeze(0)  # (1, 1, h, w)

        window_scores = F.avg_pool2d(
            sal,
            kernel_size=s,
            stride=1,
        ).squeeze(0).squeeze(0)  # (h - s + 1, w - s + 1)

        # Temperature: <1 concentrates on high-saliency patches,
        # >1 flattens the distribution.
        window_scores = torch.pow(
            window_scores.clamp(min=0.0) + 1e-12,
            1.0 / temperature,
        )

        window_probs = window_scores / window_scores.sum()


    # Validate before sampling.
    if not torch.all(torch.isfinite(window_probs)):
        raise ValueError("Non-finite values in patch probability distribution.")

    if not torch.all(window_probs >= 0):
        raise ValueError("Negative values in patch probability distribution.")

    prob_sum = window_probs.sum().item()
    if abs(prob_sum - 1.0) > 1e-6:
        raise ValueError(f"Probabilities do not sum to 1: {prob_sum}")

    flat_probs = window_probs.reshape(-1)
    flat_idx = torch.multinomial(
        flat_probs,
        num_samples=1,
        replacement=False,
    ).item()

    n_cols = window_probs.shape[1]
    center_h, center_w = divmod(flat_idx, n_cols)

    if return_prob:
        return int(center_h), int(center_w), float(flat_probs[flat_idx].item()), window_probs

    return int(center_h), int(center_w)


if __name__ == "__main__":
    # torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # image = io.read_image('data/imagenet/val/n01496331/ILSVRC2012_val_00008749.JPEG')
    image = torch.from_numpy(np.load('outputs/imagenet/query/audioclip-temp-0_1/x_inits.npy')[0])
    image = image.float()
    # Normalize to [0, 1]
    image_min = image.min()
    image_max = image.max()
    image = (image - image_min) / (image_max - image_min + 1e-8)
    image = image.to(device)
    # 0.3 and pool_size=3 are EXAMPLE choices, not recovered paper settings.
    estimator = CalculateJND(c_lt=0.3, pool_size=3, value_range=1.0).to(device)
    with torch.no_grad():
        thresholds = estimator(image)
        candidate = image + 0.5 * torch.randn_like(image)
        projected = project_to_jnd(candidate, image, thresholds, eps=16/255, gamma=1.0, value_range=255)
    
    plt.subplots(2, 3, figsize=(16, 8))
    for idx, temp in enumerate([0.1, 0.3, 0.5, 0.7, 1.0, 2.0]):
        points = []
        for i in range(10):
            # center_h, center_w = sample_patch_location(thresholds, s=16, temperature=temp)
            center_h, center_w, prob, window_probs = sample_patch_location(thresholds, s=16, temperature=temp, return_prob=True)
            points.append((center_w, center_h))
        plt.subplot(2, 3, idx + 1)
        plt.title(f"Temperature: {temp}")
        # Show original image
        plt.imshow(image.permute(1, 2, 0).cpu().numpy())
        # Overlay probability heatmap with transparency
        plt.imshow(window_probs.cpu().numpy(), cmap='viridis', alpha=0.5)
        for point in points:
            # Draw 16x16 rectangle with point as top-left corner
            rect = plt.Rectangle((point[0], point[1]), 16, 16,
                                linewidth=1, edgecolor='red', facecolor='none')
            plt.gca().add_patch(rect)
    plt.show()

    # plt.figure(figsize=(12, 10))
    # plt.subplot(2, 2, 1)
    # plt.imshow(image.permute(1, 2, 0).cpu().numpy())
    # plt.title('Original Image')
    # plt.axis('off')
    # # Calculate global min/max for consistent color scale
    # vmin = thresholds.min().item()
    # vmax = thresholds.max().item()
    # plt.subplot(2, 2, 2)
    # plt.imshow(thresholds[0, :, :].cpu().numpy(), cmap='viridis')
    # plt.title('Red Channel (JND)')
    # plt.colorbar(label='Intensity')
    # plt.subplot(2, 2, 3)
    # plt.imshow(thresholds[1, :, :].cpu().numpy(), cmap='viridis', vmin=vmin, vmax=vmax)
    # plt.title('Green Channel (JND)')
    # plt.colorbar(label='Intensity')
    # plt.subplot(2, 2, 4)
    # plt.imshow(thresholds[2, :, :].cpu().numpy(), cmap='viridis', vmin=vmin, vmax=vmax)
    # plt.title('Blue Channel (JND)')
    # plt.colorbar(label='Intensity')
    # plt.tight_layout()
    # plt.show()


    