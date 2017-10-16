# config.py ---
#
# Filename: config.py
# Description:
# Author: Kwang Moo Yi
# Maintainer:
# Created: Thu Jul  6 15:39:25 2017 (+0200)
# Version:
# Package-Requires: ()
# URL:
# Doc URL:
# Keywords:
# Compatibility:
#
#

# Commentary:
#
#
#
#

# Change Log:
#
#
#
# Copyright (C), EPFL Computer Vision Lab.

# Code:

import numpy as np


def get_ratio_scale(config):
    """Calculate ratio_scale from other configs"""

    kp_input_size = config.kp_input_size
    kp_base_scale = config.kp_base_scale
    ratio_scale = (float(kp_input_size) / 2.0) / kp_base_scale

    return ratio_scale


def get_patch_size_no_aug(config):
    """Determine large patch size without rotation augmentations"""

    desc_input_size = config.desc_input_size
    desc_support_ratio = config.desc_support_ratio

    ratio_scale = get_ratio_scale(config)
    patch_size = np.round(
        float(desc_input_size) * ratio_scale / desc_support_ratio)

    return patch_size


def get_patch_size(config):
    """Get the large patch size from other configs"""

    patch_size = get_patch_size_no_aug(config)
    if config.use_augmented_set:
        patch_size = np.ceil(np.sqrt(2) * patch_size)

    return patch_size


#
# config.py ends here
