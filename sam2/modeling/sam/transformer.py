
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import math
import warnings
from functools import partial
from typing import Tuple, Type

import torch
import torch.nn.functional as F
from torch import nn, Tensor

from sam2.modeling.position_encoding import apply_rotary_enc, compute_axial_cis
from sam2.modeling.sam2_utils import MLP
from sam2.utils.misc import get_sdpa_settings

warnings.simplefilter(action="ignore", category=FutureWarning)
# Check whether Flash Attention is available (and use it by default)
OLD_GPU, USE_FLASH_ATTN, MATH_KERNEL_ON = get_sdpa_settings()