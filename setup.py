# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# Package metadata
NAME = "SAM 2"
VERSION = "1.0"
DESCRIPTION = "SAM 2: Segment Anything in Im