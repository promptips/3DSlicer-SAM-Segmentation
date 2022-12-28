
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from functools import partial
from typing import List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from sam2.modeling.backbones.utils import (
    PatchEmbed,
    window_partition,
    window_unpartition,
)

from sam2.modeling.sam2_utils import DropPath, MLP


def do_pool(x: torch.Tensor, pool: nn.Module, norm: nn.Module = None) -> torch.Tensor:
    if pool is None:
        return x
    # (B, H, W, C) -> (B, C, H, W)
    x = x.permute(0, 3, 1, 2)
    x = pool(x)
    # (B, C, H', W') -> (B, H', W', C)
    x = x.permute(0, 2, 3, 1)
    if norm:
        x = norm(x)

    return x


class MultiScaleAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_out: int,
        num_heads: int,
        q_pool: nn.Module = None,
    ):
        super().__init__()

        self.dim = dim
        self.dim_out = dim_out
        self.num_heads = num_heads
        self.q_pool = q_pool
        self.qkv = nn.Linear(dim, dim_out * 3)
        self.proj = nn.Linear(dim_out, dim_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, _ = x.shape
        # qkv with shape (B, H * W, 3, nHead, C)
        qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1)
        # q, k, v with shape (B, H * W, nheads, C)
        q, k, v = torch.unbind(qkv, 2)

        # Q pooling (for downsample at stage changes)
        if self.q_pool: