# wrapper.py ---
#
# Filename: wrapper.py
# Description:
# Author: Kwang Moo Yi
# Maintainer:
# Created: Thu Jun 29 14:55:21 2017 (+0200)
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

import datasets.eccv2016.eccv as old_impl
from datasets.eccv2016.custom_types import paramStruct
from utils import get_patch_size, get_patch_size_no_aug, get_ratio_scale


def config_to_param(config):
    """The function that takes care of the transfer to the new framework"""

    param = paramStruct()

    # Param Group "dataset"
    param.dataset.nTestPercent = int(20)
    param.dataset.dataType = "ECCV"
    param.dataset.nValidPercent = int(20)
    param.dataset.fMinKpSize = float(2.0)
    param.dataset.nPosPerImg = int(-1)
    # Note that we are passing a list. This module actually supports
    # concatenating datsets.
    param.dataset.trainSetList = ["ECCV/" + config.data_name]
    param.dataset.nNegPerImg = int(1000)
    param.dataset.nTrainPercent = int(60)

    # Param Group "patch"
    if config.old_data_compat:
        param.patch.nPatchSize = int(get_patch_size(config))
    else:
        param.patch.nPatchSize = int(get_patch_size_no_aug(config))
        param.patch.nPatchSizeAug = int(get_patch_size(config))
    param.patch.noscale = False
    param.patch.fNegOverlapTh = float(0.1)
    param.patch.sNegMineMethod = "use_all_SIFT_points"
    param.patch.fRatioScale = float(get_ratio_scale(config))
    param.patch.fPerturbInfo = np.array([0.2, 0.2, 0.0]).astype(float)
    if config.old_data_compat:
        param.patch.nMaxRandomNegMineIter = int(500)
    else:
        param.patch.nMaxRandomNegMineIter = int(100)
    param.patch.fMaxScale = 1.0
    param.patch.bPerturb = 1.0

    # Param Group "model"
    param.model.nDescInputSize = int(config.desc_input_size)

    # override folders from config
    setattr(param, "data_dir", config.data_dir)
    setattr(param, "temp_dir", config.temp_dir)
    setattr(param, "scratch_dir", config.scratch_dir)

    return param


class Wrapper(object):
    """The wrapper class for eccv data.

    Since we want to re-use the code for loading and storing data, but do *not*
    want to bother integrating it into the new framework, we simply create a
    wrapper on top, which translates the old format to the new one. This is
    likely to cause some unecessary overhead, but is not so critical right now.

    The role of this class is to:

    * Create a dictionary that allows easy access to raw data

    """

    def __init__(self, config, rng):

        # Placeholder for the data dictionary
        self.data = {}

        # Use the old  data module to load data. Processing data loading for
        # differen tasks
        for task in ["train", "valid", "test"]:
            param = config_to_param(config)
            old_data = old_impl.data_obj(param, task)
            # Some sanity check to make sure that the data module is behaving
            # as intended.
            assert old_data.patch_height == old_data.patch_width
            assert old_data.patch_height == get_patch_size(config)
            assert old_data.num_channel == config.nchannel
            assert old_data.out_dim == config.nchannel
            self.data[task] = {
                "patch": old_data.x,
                "xyz": old_data.pos,
                "angle": old_data.angle.reshape(-1, 1),
                "ID": old_data.ID,
            }

        # data ordering of this class
        self.data_order = "NCHW"

        # Save the hash, for the pairs folder
        self.hash = old_data.hash
#
# wrapper.py ends here
