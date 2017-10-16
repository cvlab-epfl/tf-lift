# train.py ---
#
# Filename: train.py
# Description:
# Author: Kwang Moo Yi
# Maintainer:
# Created: Thu Jul  6 16:23:31 2017 (+0200)
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


def get_hard_batch(loss_list, data_list):
    """Returns a batch with the hardest samples.

    Given a list of losses and data, merges them together to form a batch with
    is the hardest samples.

    NOTE
    ----

    data item in the data_list is accessible by data_list[i]["patch"]["P1"][j]

    loss item in the loss_list is accessible by loss_list[i][j]

    """

    # Concatenate all data
    all_data = {}
    for _type in data_list[0]:
        all_data[_type] = {}
        for _name in data_list[0][_type]:
            # "aug_rot" is a dictionary with two keys
            if isinstance(data_list[0][_type][_name], dict):
                all_data[_type][_name] = {}
                for _key in data_list[0][_type][_name]:
                    all_data[_type][_name][_key] = np.concatenate(
                        (lambda _t=_type, _n=_name, _k=_key, _d=data_list: [
                            data[_t][_n][_k] for data in _d])(), axis=0)
            else:
                all_data[_type][_name] = np.concatenate(
                    (lambda _t=_type, _n=_name, _d=data_list: [
                        data[_t][_n] for data in _d])(), axis=0)

    # Extract batch size
    batch_size = len(loss_list[0])

    # Concatenate all loss
    all_loss = np.concatenate(loss_list, axis=0)

    # Sort by loss and get indices for the hardes ones
    ind = np.argsort(all_loss)[::-1]
    ind = ind[:batch_size]

    # Select the hardest examples
    for _type in all_data:
        for _name in all_data[_type]:
            if isinstance(all_data[_type][_name], dict):
                all_data[_type][_name] = {
                    _key: all_data[_type][_name][_key][ind]
                    for _key in all_data[_type][_name]}
            else:
                all_data[_type][_name] = all_data[_type][_name][ind]

    return all_data


#
# train.py ends here
