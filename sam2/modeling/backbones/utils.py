
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Some utilities for backbones, in particular for windowing"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def window_partition(x, window_size):
    """
    Partition into non-overlapping windows with padding if needed.
    Args: