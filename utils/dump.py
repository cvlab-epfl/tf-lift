# dump.py ---
#
# Filename: dump.py
# Description:
# Author: Kwang Moo Yi
# Maintainer:
# Created: Thu Jul  6 15:36:36 2017 (+0200)
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

import os

import h5py
import tensorflow as tf

from utils.legacy import load_legacy_network

# Some constant strings
best_val_loss_filename = "best_val_loss.h5"
best_step_filename = "step.h5"


def saveh5(dict_to_dump, dump_file_full_name):
    ''' Saves a dictionary as h5 file '''

    with h5py.File(dump_file_full_name, 'w') as h5file:
        if isinstance(dict_to_dump, list):
            for i, d in enumerate(dict_to_dump):
                newdict = {'dict' + str(i): d}
                writeh5(newdict, h5file)
        else:
            writeh5(dict_to_dump, h5file)


def writeh5(dict_to_dump, h5node):
    ''' Recursive function to write dictionary to h5 nodes '''

    for _key in dict_to_dump.keys():
        if isinstance(dict_to_dump[_key], dict):
            h5node.create_group(_key)
            cur_grp = h5node[_key]
            writeh5(dict_to_dump[_key], cur_grp)
        else:
            h5node[_key] = dict_to_dump[_key]


def loadh5(dump_file_full_name):
    ''' Loads a h5 file as dictionary '''

    with h5py.File(dump_file_full_name, 'r') as h5file:
        dict_from_file = readh5(h5file)

    return dict_from_file


def readh5(h5node):
    ''' Recursive function to read h5 nodes as dictionary '''

    dict_from_file = {}
    for _key in h5node.keys():
        if isinstance(h5node[_key], h5py._hl.group.Group):
            dict_from_file[_key] = readh5(h5node[_key])
        else:
            dict_from_file[_key] = h5node[_key].value

    return dict_from_file


def save_network(supervisor, subtask, verbose=True):
    """Save the current training status"""

    # Skip if there's no saver of this subtask
    if subtask not in supervisor.saver:
        return

    cur_logdir = os.path.join(supervisor.config.logdir, subtask)
    # Create save directory if it does not exist
    if verbose:
        print("")
        print("[{}] Checking if save directory exists in {}"
              "".format(subtask, cur_logdir))
    if not os.path.exists(cur_logdir):
        os.makedirs(cur_logdir)
    # Save the model
    supervisor.saver[subtask].save(
        supervisor.sess,
        os.path.join(cur_logdir, "network")
    )
    if verbose:
        print("[{}] Saved model at {}"
              "".format(subtask, cur_logdir))
    # Save mean std
    saveh5(supervisor.network.mean, os.path.join(cur_logdir, "mean.h5"))
    saveh5(supervisor.network.std, os.path.join(cur_logdir, "std.h5"))
    if verbose:
        print("[{}] Saved input normalization at {}"
              "".format(subtask, cur_logdir))
    # Save the validation loss
    saveh5({subtask: supervisor.best_val_loss[subtask]},
           os.path.join(cur_logdir, best_val_loss_filename))
    if verbose:
        print("[{}] Saved best validation at {}"
              "".format(subtask, cur_logdir))
    # Save the step
    saveh5({subtask: supervisor.best_step[subtask]},
           os.path.join(cur_logdir, best_step_filename))
    if verbose:
        print("[{}] Saved best step at {}"
              "".format(subtask, cur_logdir))

    # We also save the network metadata here
    supervisor.saver[subtask].export_meta_graph(
        os.path.join(cur_logdir, "network.meta"))


def restore_network(supervisor, subtask):
    """Restore training status"""

    # Skip if there's no saver of this subtask
    if subtask not in supervisor.saver:
        return False

    is_loaded = False

    # Check if pretrain weight file is specified
    predir = getattr(supervisor.config, "pretrained_{}".format(subtask))
    # Try loading the old weights
    is_loaded += load_legacy_network(supervisor, subtask, predir)
    # Try loading the tensorflow weights
    is_loaded += load_network(supervisor, subtask, predir)

    # Load network using tensorflow saver
    logdir = os.path.join(supervisor.config.logdir, subtask)
    is_loaded += load_network(supervisor, subtask, logdir)

    return is_loaded


def load_network(supervisor, subtask, load_dir):
    """Load function for our new framework"""

    print("[{}] Checking if previous Tensorflow run exists in {}"
          "".format(subtask, load_dir))
    latest_checkpoint = tf.train.latest_checkpoint(load_dir)
    if latest_checkpoint is not None:
        # Load parameters
        supervisor.saver[subtask].restore(
            supervisor.sess,
            latest_checkpoint
        )
        print("[{}] Loaded previously trained weights".format(subtask))
        # Save mean std (falls back to default if non-existent)
        if os.path.exists(os.path.join(load_dir, "mean.h5")):
            supervisor.network.mean = loadh5(os.path.join(load_dir, "mean.h5"))
            supervisor.network.std = loadh5(os.path.join(load_dir, "std.h5"))
            print("[{}] Loaded input normalizers".format(subtask))
        # Load best validation result
        supervisor.best_val_loss[subtask] = loadh5(
            os.path.join(load_dir, best_val_loss_filename)
        )[subtask]
        print("[{}] Loaded best validation result = {}".format(
            subtask, supervisor.best_val_loss[subtask]))
        # Load best validation result
        supervisor.best_step[subtask] = loadh5(
            os.path.join(load_dir, best_step_filename)
        )[subtask]
        print("[{}] Loaded best step = {}".format(
            subtask, supervisor.best_step[subtask]))

        return True

    else:
        print("[{}] No previous Tensorflow result".format(subtask))

        return False


#
# dump.py ends here
