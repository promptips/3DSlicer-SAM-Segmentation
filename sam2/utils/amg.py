# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
from copy import deepcopy
from itertools import product
from typing import Any, Dict, Generator, ItemsView, List, Tuple

import numpy as np
import torch

# Very lightly adap