import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import torch

from addict import Dict
from torch.distributions import StudentT




def lower_half_to_task(img, t_noise=None, device=None):
    B, C, H, W = img.shape
    assert H % 2 == 0, "Image height must be even to split into upper and lower halves."

    img = img.view(B, C, -1)

    if t_noise is not None:
        if t_noise == -1:
            t_noise = 0.09 * torch.rand(img.shape)
        img += t_noise * StudentT(2.1).rsample(img.shape)

    device = img.device if device is None else device

    batch = Dict()

    num_ctx = (H // 2) * W
    idxs_ctx = torch.arange(num_ctx, H * W).to(img.device).unsqueeze(0).repeat(B, 1)
    x1_ctx, x2_ctx = idxs_ctx // W, idxs_ctx % W
    batch.xc = torch.stack([
        2 * x1_ctx.float() / (H - 1) - 1,
        2 * x2_ctx.float() / (W - 1) - 1], -1).to(device)
    batch.yc = (torch.gather(img, -1, idxs_ctx.unsqueeze(1).repeat(1, C, 1))
                .transpose(1, 2) - 0.5).to(device)

    num_tar = num_ctx
    idxs_tar = torch.arange(num_tar).to(img.device).unsqueeze(0).repeat(B, 1)
    x1_tar, x2_tar = idxs_tar // W, idxs_tar % W
    batch.xt = torch.stack([
        2 * x1_tar.float() / (H - 1) - 1,
        2 * x2_tar.float() / (W - 1) - 1], -1).to(device)
    batch.yt = (torch.gather(img, -1, idxs_tar.unsqueeze(1).repeat(1, C, 1))
                .transpose(1, 2) - 0.5).to(device)

    batch.x = torch.cat([batch.xc, batch.xt], dim=1)
    batch.y = torch.cat([batch.yc, batch.yt], dim=1)

    return batch

def upper_half_to_task(img, t_noise=None, device=None):
    B, C, H, W = img.shape
    assert H % 2 == 0, "Image height must be even to split into upper and lower halves."

    img = img.view(B, C, -1)

    if t_noise is not None:
        if t_noise == -1:
            t_noise = 0.09 * torch.rand(img.shape)
        img += t_noise * StudentT(2.1).rsample(img.shape)

    device = img.device if device is None else device

    batch = Dict()

    num_ctx = (H // 2) * W
    idxs_ctx = torch.arange(num_ctx).to(img.device).unsqueeze(0).repeat(B, 1)
    x1_ctx, x2_ctx = idxs_ctx // W, idxs_ctx % W
    batch.xc = torch.stack([
        2 * x1_ctx.float() / (H - 1) - 1,
        2 * x2_ctx.float() / (W - 1) - 1], -1).to(device)
    batch.yc = (torch.gather(img, -1, idxs_ctx.unsqueeze(1).repeat(1, C, 1))
                .transpose(1, 2) - 0.5).to(device)

    num_tar = num_ctx
    idxs_tar = torch.arange(num_tar, 2 * num_tar).to(img.device).unsqueeze(0).repeat(B, 1)
    x1_tar, x2_tar = idxs_tar // W, idxs_tar % W
    batch.xt = torch.stack([
        2 * x1_tar.float() / (H - 1) - 1,
        2 * x2_tar.float() / (W - 1) - 1], -1).to(device)
    batch.yt = (torch.gather(img, -1, idxs_tar.unsqueeze(1).repeat(1, C, 1))
                .transpose(1, 2) - 0.5).to(device)

    batch.x = torch.cat([batch.xc, batch.xt], dim=1)
    batch.y = torch.cat([batch.yc, batch.yt], dim=1)

    return batch