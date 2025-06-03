# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------

import random
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

try:
    from scipy.ndimage import binary_erosion
except ImportError:
    pass  # Don't fail if scipy is not installed. It's only necessary for this one file.


def generate_colormap(N: int, seed: int = 0) -> List[Tuple[float, float, float]]:
    """Generates a equidistant colormap with N elements."""
    random.seed(seed)

    def generate_color():
        return (random.random(), random.random(), random.random())

    return [generate_color() for _ in range(N)]


def make_visualization(
    img: Image, source: torch.Tensor, patch_size: int = 16, class_token: bool = True
) -> Image:
    """
    Create a visualization like in the paper.

    Args:
     -

    Returns:
     - A PIL image the same size as the input.
    """

    img = np.array(img.convert("RGB")) / 255.0
    source = source.detach().cpu()

    h, w, _ = img.shape
    ph = h // patch_size
    pw = w // patch_size

    if class_token:
        source = source[:, :, 1:]

    vis = source.argmax(dim=1)
    num_groups = vis.max().item() + 1

    cmap = generate_colormap(num_groups)
    vis_img = 0

    for i in range(num_groups):
        mask = (vis == i).float().view(1, 1, ph, pw)
        mask = F.interpolate(mask, size=(h, w), mode="nearest")
        mask = mask.view(h, w, 1).numpy()

        color = (mask * img).sum(axis=(0, 1)) / mask.sum()
        mask_eroded = binary_erosion(mask[..., 0])[..., None]
        mask_edge = mask - mask_eroded

        if not np.isfinite(color).all():
            color = np.zeros(3)

        vis_img = vis_img + mask_eroded * color.reshape(1, 1, 3)
        vis_img = vis_img + mask_edge * np.array(cmap[i]).reshape(1, 1, 3)

    # Convert back into a PIL image
    vis_img = Image.fromarray(np.uint8(vis_img * 255))

    return vis_img

def make_spatial_video_visualization(
    video: torch.Tensor, source: torch.Tensor, patch_size: Tuple[int, int, int] = (16, 16, 2), class_token: bool = True,
    average_colour = True) -> np.ndarray:
    video = torch.permute(video, (0, 2, 3, 1)).numpy()
    source = source.detach().cpu()

    t, h, w, _ = video.shape

    if class_token:
        source = source[:, :, 1:]

    vis_vid = []

    for frame_idx in range(0, t, patch_size[2]):

        local_t = patch_size[2]

        ph = h // patch_size[0]
        pw = w // patch_size[1]
        pt = 1

        vis = source[frame_idx // patch_size[2]][None].argmax(dim=1)
        num_groups = vis.max().item() + 1

        cmap = generate_colormap(num_groups)
        vis_img = 0

        for i in range(num_groups):
            mask = (vis == i).float().view(1, 1, pt, ph, pw)
            mask = F.interpolate(mask, size=(local_t, h, w), mode="nearest")
            mask = mask.view(local_t, h, w, 1).numpy()

            color = (mask * video[frame_idx:frame_idx+patch_size[2]]).sum(axis=(0, 1, 2)) / mask.sum()
            mask_eroded = np.stack(list(map(lambda x: binary_erosion(x[..., 0]), mask)))[..., None]
            mask_edge = mask - mask_eroded

            if not np.isfinite(color).all():
                color = np.zeros(3)

            if average_colour:
                vis_img = vis_img + mask_eroded * color.reshape(1, 1, 1, 3)
            else:
                vis_img = vis_img + mask_eroded * video[frame_idx:frame_idx+patch_size[2]]
            vis_img = vis_img + mask_edge * np.array(cmap[i]).reshape(1, 1, 1, 3)

        vis_vid.append(vis_img)

    vis_vid = np.uint8(np.concatenate(vis_vid) * 255)

    return vis_vid

def make_spatiotemporal_video_visualization(
    video: torch.Tensor, source: torch.Tensor, patch_size: Tuple[int, int, int] = (16, 16, 2), class_token: bool = True,
    average_colour = True, separate = False) -> Tuple[np.ndarray, List]:
    video = torch.permute(video, (0, 2, 3, 1)).numpy()
    source = source.detach().cpu()

    t, h, w, _ = video.shape
    ph = h // patch_size[0]
    pw = w // patch_size[1]
    pt = t // patch_size[2]

    if class_token:
        source = source[:, :, 1:]

    vis = source.argmax(dim=1)
    num_groups = vis.max().item() + 1
    vis[source.sum(dim=1) == 0] = num_groups

    cmap = generate_colormap(num_groups)
    vis_vid = 0
    separate_tokens = []

    for i in range(num_groups):
        mask = (vis == i).float().view(1, 1, pt, ph, pw)
        mask = F.interpolate(mask, size=(t, h, w), mode="nearest")
        mask = mask.view(t, h, w, 1).numpy()

        color = (mask * video).sum(axis=(0, 1, 2)) / mask.sum()
        mask_eroded = np.stack(list(map(lambda x: binary_erosion(x[..., 0]), mask)))[..., None]
        mask_edge = mask - mask_eroded

        if not np.isfinite(color).all():
            color = np.zeros(3)

        if average_colour:
            token = mask_eroded * color.reshape(1, 1, 1, 3)
        else:
            token = mask_eroded * video
        vis_vid = vis_vid + token
        vis_vid = vis_vid + mask_edge * np.array(cmap[i]).reshape(1, 1, 1, 3)

        if separate:
            separate_tokens.append(np.uint8(token * 225))

    vis_vid = np.uint8(vis_vid * 255)

    return vis_vid, separate_tokens

def concatenate_images(video: np.ndarray, ncols=8, nrows=4) -> Image:
    n, h, w, c = video.shape
    concatenated_image = Image.new('RGB', (w * ncols, h * nrows))
    for y in range(nrows):
        for x in range(ncols):
            concatenated_image.paste(Image.fromarray(video[y * ncols + x]), (x * w, y * h))

    return concatenated_image
