# lift.py ---
#
# Filename: lift.py
# Description:
# Author: Kwang Moo Yi
# Maintainer:
# Created: Wed Jul  5 19:29:17 2017 (+0200)
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


from __future__ import print_function

import importlib
import os
import sys

import numpy as np
import h5py

from six.moves import xrange
from utils import loadh5, saveh5


class Dataset(object):
    """The Dataset Module

    LATER: For validation data, next_batch should also consider the pos-to-neg
           ratio. For now, we just weight them

    LATER: Smarter cache system
    LATER: Optimize LUT generation

    """

    def __init__(self, config, rng):
        self.config = config
        self.rng = rng

        # Hard-code the 3D distance threshold
        # 3D points can be too close to be non-matches
        if self.config.data_type == "chalmers":
            if self.config.data_name == "oxford":
                self.th_dist = 2e-3
            elif self.config.data_name == "oxford2":
                self.th_dist = 5e-3
            else:
                raise RuntimeError(
                    "Unknown data_name: '{}'".format(
                        self.config.data_name))

        # Look Up Tables
        self.LUT_kp = {}
        self.LUT_nonkp = {}
        if hasattr(self, "th_dist"):
            self.LUT_below_th = {}

        # Initialize the appropriate data module
        self.data_module = importlib.import_module(
            "datasets.{}.wrapper".format(self.config.data_type))

        # Initialize the wrapper class. This should also make the data
        # available on demand. For now, we just get the handles to the data
        # stored in h5 files since it's too large
        self.data_wrapper = self.data_module.Wrapper(config, rng)
        # Alias data for easy access
        self.data = self.data_wrapper.data

        # Overwrite pairs directory with dataset name
        self.pair_dir = '{}/{}/{}/'.format(
            config.pair_dir, config.data_type, config.data_name)
        # And with the hash
        self.pair_dir = '{}/{}'.format(
            self.pair_dir, self.data_wrapper.hash)

        # Pointers for each task
        self.batch_start = {}
        self.epoch = {}
        self.ind = {}
        self.pairs = {}
        for task in self.data:
            # Counters to indicate the position of the batch in epoch
            self.batch_start[task] = 0
            self.epoch[task] = 0

            # Mark LUT to un-init
            self.LUT_kp[task] = None
            self.LUT_nonkp[task] = None
            if hasattr(self, "th_dist"):
                self.LUT_below_th[task] = None

            # Create initial pairs
            self.pairs[task] = self._create_pairs(
                task, self.config.pair_use_cache)
            # Create Initial random permutation
            self.ind[task] = np.random.permutation(
                len(self.pairs[task]["P1"]))

    def next_batch(self, task, subtask, batch_size, aug_rot=0, num_empty=0):

        # Dictionary to return
        cur_data = {}

        assert batch_size > 0

        # Check if we can fill the data
        if self.batch_start[task] + batch_size >= len(self.ind[task]):
            # If we can't shuffle the indices, reset pointer, and increase
            # epoch
            self.ind[task] = np.random.permutation(
                len(self.pairs[task]["P1"]))
            self.batch_start[task] = 0
            self.epoch[task] += 1
            # Every N epochs, recreate pairs. Note that the number of pairs
            # will NEVER change
            if self.config.pair_interval > 0 and task == "train":
                # Always active for oxford (small set)
                if self.config.data_name is "oxford" or self.config.regen_pairs:
                    if self.epoch[task] % self.config.pair_interval == 0:
                        self.pairs[task] = self._create_pairs(
                            task, self.config.pair_use_cache, overwrite=False)

        # Pack the data according to pairs
        for _type in ["patch", "xyz", "angle"]:
            cur_data[_type] = {}
            for _name in ["P1", "P2", "P3", "P4"]:
                _pair = self.pairs[task][_name]
                cur_data[_type][_name] = np.asarray([
                    self.data[task][_type][_pair[
                        self.ind[task][_i + self.batch_start[task]]
                    ]] for _i in xrange(batch_size)
                ])

                # Work-around for partial batches: add empty samples
                if num_empty > 0:
                    cur_data[_type][_name].resize(
                        (cur_data[_type][_name].shape[0] + num_empty,) +
                        cur_data[_type][_name].shape[1:])

                # If it's the patch, make it into NHWC format
                if _type == "patch":
                    if self.data_wrapper.data_order == "NCHW":
                        cur_data[_type][_name] = np.array(
                            np.transpose(cur_data[_type][_name],
                                         (0, 2, 3, 1))
                        )

        # Augment rotations: -1 (zeros), 0 (no aug ST, do nothing), 1 (random)
        # Allows us to separate train/val (not doing it for now)
        if aug_rot > 0:
            cur_data["aug_rot"] = {}
            for _name in ["P1", "P2", "P3", "P4"]:
                rot = 2 * np.pi * np.random.rand(batch_size)
                cur_data["aug_rot"][_name] = {}
                cur_data["aug_rot"][_name]["angle"] = rot
                cur_data["aug_rot"][_name]["cs"] = np.concatenate(
                    (np.cos(rot)[..., None], np.sin(rot)[..., None]),
                    axis=1)
        elif aug_rot < 0:
            cur_data["aug_rot"] = {}
            for _name in ["P1", "P2", "P3", "P4"]:
                rot = np.zeros(batch_size)
                cur_data["aug_rot"][_name] = {}
                cur_data["aug_rot"][_name]["angle"] = rot
                cur_data["aug_rot"][_name]["cs"] = np.concatenate(
                    (np.cos(rot)[..., None], np.sin(rot)[..., None]),
                    axis=1)

        # Update the batch start location
        self.batch_start[task] += batch_size

        return cur_data

    def _create_pairs(self, task, use_cache=True, overwrite=False):
        """Generate the pairs from ID

        This function should return the pair indices given the task. The return
        type is expected to be a dictionary, where you can access by doing
        res["P1"], for example.

        """

        # Use the cache file if asked
        if use_cache:
            pairs_file = os.path.join(
                self.pair_dir, "{}.h5".format(task))
            if not os.path.exists(self.pair_dir):
                os.makedirs(self.pair_dir)
            if os.path.exists(pairs_file) and not overwrite:
                if self.config.use_augmented_set:
                    # Add rotation augmentation if it does not exist (compat)
                    _f = h5py.File(pairs_file, "r+")
                    for name in ["P1", "P2", "P3", "P4"]:
                        if (name + "_rot_aug") not in _f:
                            _f.create_dataset(
                                name + "_rot_aug", data=2 *
                                np.pi * self.rng.rand(_f[name].size))
                    _f.close()
                return loadh5(pairs_file)

        # Make a lookup table for getting the indices of each sample. This can
        # be easily done by sorting by ID, and storing the indices to a
        # dictionary
        if self.LUT_kp[task] is None:
            # Create empty dictionary
            self.LUT_kp[task] = {}
            self.LUT_nonkp[task] = {}
            if hasattr(self, "th_dist"):
                if self.data[task]["dist_idx"]:
                    self.LUT_below_th[task] = {}
                    # Build list of indices
                    dist_idx_reverse = np.zeros(
                        self.data[task]["ID"].max() + 1,
                        dtype=np.int64)
                    dist_idx_reverse[self.data[task]["dist_idx"]] = range(
                        1, self.data[task]["dist_idx"].size + 1)
                    dist_idx_reverse -= 1
                    self.has_distmat = True
                else:
                    self.has_distmat = False

                # Retrieve coordinates if necessary
                if not self.has_distmat:
                    if not hasattr(self, 'points'):
                        self.points = {}
                    fn = self.config.data_dir + '/' + \
                        self.config.data_type + '/' + \
                        self.config.data_name + '/' + \
                        self.config.data_name + '.h5'
                    with h5py.File(fn, 'r') as f:
                        self.points = f['points'].value

            # Argsort the ID
            print("[{}] sorting IDs...".format(task))
            ID = self.data[task]["ID"]
            ind = np.argsort(ID)
            # Go through them one by one -- the easy way
            print("[{}] building LUT...".format(task))
            start = 0
            start_ID = ID[ind[start]]
            for end in xrange(1, len(ind)):
                if end % 100 == 0:
                    print("\r --- working on {}/{}".format(
                        end + 1, len(ind)), end="")
                    sys.stdout.flush()

                # Check if indices have changed or we reached the end
                if start_ID != ID[ind[end]] or end == len(ind) - 1:
                    # Store them in proper LUT
                    if start_ID < 0:
                        self.LUT_nonkp[task][-1] = ind[start:end]
                    else:
                        self.LUT_kp[task][start_ID] = ind[start:end]
                        if hasattr(self, "th_dist"):
                            if self.has_distmat:
                                v = dist_idx_reverse[start_ID]
                                assert v >= 0, "Invalid index"
                                d = self.data[task]["dist_mat"][v]
                                # 3D points are 1-indexed
                                self.LUT_below_th[task][start_ID] = 1 + \
                                    np.where(d > self.th_dist)[0]
                    # Update start position
                    start = end
                    start_ID = ID[ind[start]]
            print("\r --- done.                              ")

        # The dictionary to return
        cur_pairs = {}
        for name in ["P1", "P2", "P3", "P4"]:
            cur_pairs[name] = []

        # For each kp item, create a random pair
        #
        # Note that this is different from what we used to do. This way seems
        # better as we will look at each keypoint once.
        print("[{}] creating pairs...".format(task))
        num_kp = len(self.LUT_kp[task])
        for i in xrange(num_kp):
            if i % 100 == 0:
                print("\r --- working on {}/{}".format(
                    i + 1, num_kp), end="")
                sys.stdout.flush()
            _kp = list(self.LUT_kp[task].keys())[i]

            # Check if we have enough views of this point and skip
            if len(self.LUT_kp[task][_kp]) < 2:
                continue

            # For P1 and P2 -- Select two random patches
            P1, P2 = self.rng.choice(
                self.LUT_kp[task][_kp], 2, replace=False)

            # For P3 -- Select a different keypoint randomly
            # Generate a list of keys
            valid_keys = set(self.LUT_kp[task])
            # Remove self
            valid_keys -= set([_kp])
            # Remove points below the distance threshold
            if hasattr(self, "th_dist"):
                if self.has_distmat:
                    valid_keys -= set(valid_keys.intersection(
                        self.LUT_below_th[task][_kp]))
                else:
                    # Recompute distances for every 3D point
                    xyz = self.points[_kp - 1]
                    d_sq = np.sum((xyz[None, ...].repeat(self.points.shape[
                        0], axis=0) - self.points) ** 2, axis=1)
                    valid_keys -= set(1 + np.where(d_sq < self.th_dist**2)[0])
                    if len(valid_keys) == 0:
                        print('Should never be here')
                        import IPython
                        IPython.embed()

            _kp2 = self.rng.choice(list(valid_keys))
            P3 = self.rng.choice(self.LUT_kp[task][_kp2], 1)[0]

            # For P4 -- Select a random nonkp
            P4 = self.rng.choice(self.LUT_nonkp[task][-1], 1)[0]

            # Append to the list
            for name in ["P1", "P2", "P3", "P4"]:
                cur_pairs[name].append(eval(name))
        print("\r --- done.                              ")

        # Convert them to np arrays
        for name in ["P1", "P2", "P3", "P4"]:
            cur_pairs[name] = np.asarray(cur_pairs[name])

        # Augment pairs with rotation data
        if self.config.use_augmented_set:
            for name in ["P1", "P2", "P3", "P4"]:
                cur_pairs[name + "_rot_aug"] = 2 * np.pi * \
                        self.rng.rand(cur_pairs[name].size)

        # Save to cache if asked
        if use_cache:
            if not os.path.exists(self.pair_dir):
                os.makedirs(self.pair_dir)
            saveh5(cur_pairs, pairs_file)

        return cur_pairs

#
# lift.py ends here
