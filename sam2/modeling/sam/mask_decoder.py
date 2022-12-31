# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional, Tuple, Type

import torch
from torch import nn

from sam2.modeling.sam2_utils import LayerNorm2d, MLP


class MaskDecoder(nn.Module):
    def _