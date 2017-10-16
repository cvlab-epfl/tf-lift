# legacy.py ---
#
# Filename: legacy.py
# Description: Functions related to using the old framework
# Author: Kwang Moo Yi
# Maintainer:
# Created: Tue Jul 11 14:52:25 2017 (+0200)
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

import tensorflow as tf


def build_legacy(network):
    """Builds the tensorflow ops related to loading legacy weights.

    Note that implementation for this function is **intentionally** left out
    from the original class as it is only a hack to allow old weights to be
    loaded into tensorflow.

    """

    # Lazy import to prevent import issues
    from utils.tf import get_tensor_shape

    # Check the current subtask
    subtask = network.config.subtask

    # Mapping from the old framework variable name to the new tensorflow
    # variable names
    name_map = {}
    if subtask == "kp" or subtask == "joint":
        # Load the keypoint only when necessary
        name_map["kp-c0"] = "network/lift/kp/conv-ghh-1"
    if subtask == "ori" or subtask == "joint":
        # Load the orientation only when necessary
        name_map["ori-c0"] = "network/lift/ori/conv-act-pool-1"
        name_map["ori-c1"] = "network/lift/ori/conv-act-pool-2"
        name_map["ori-c2"] = "network/lift/ori/conv-act-pool-3"
        name_map["ori-f3"] = "network/lift/ori/fc-ghh-drop-4"
        name_map["ori-f4"] = "network/lift/ori/fc-ghh-5"
    if subtask != "kp":
        # Load the descriptor only when necessary
        name_map["desc-1-conv"] = "network/lift/desc/conv-act-pool-norm-1"
        name_map["desc-5-conv"] = "network/lift/desc/conv-act-pool-norm-2"
        name_map["desc-9-conv"] = "network/lift/desc/conv-act-pool-3"

    # # Save name_map
    # network.legacy_name_map = name_map

    # Build placeholders from the original variable shapes and create assign
    # ops related to it
    network.legacy_weights_in = {}
    network.legacy_assign_op = {}
    # _old: variable scope name for the old framework
    # _new: variable scope name for the new framework
    for _old in name_map:
        _new = name_map[_old]
        for _param_name in ["/weights", "/biases"]:
            with tf.variable_scope("", reuse=True):
                cur_param = tf.get_variable(_new + _param_name)
            network.legacy_weights_in[_new + _param_name] = tf.placeholder(
                tf.float32,
                get_tensor_shape(cur_param),
            )
            network.legacy_assign_op[_new + _param_name] = tf.assign(
                cur_param,
                network.legacy_weights_in[_new + _param_name],
            )

    # Create function attribute that actually runs the loader with the session
    if not hasattr(network, "legacy_load_func"):
        network.legacy_load_func = {}

    def legacy_load_func(sess, model):
        # Create a feed_dict
        feed_dict = create_feed_dict(
            model, network.legacy_weights_in, name_map)
        # Run all assign ops within the session
        sess.run(network.legacy_assign_op, feed_dict=feed_dict)

    network.legacy_load_func[subtask] = legacy_load_func


def create_feed_dict(model, placeholders, name_map):
    """Creates a feed dict to use for the assignment op.

    model: dictionary containing the old weights

    placeholders: placeholders to feed to

    name_map: mapping for the old name to the new variable name

    """

    feed_dict = {}

    # For each variables we want to assign
    for _old in name_map:

        # Get the new name
        _new = name_map[_old]

        # For weights
        cur_weights = model[_old][_old + ".W"]
        # If it is 4D+, shrink dimension, as the last dimension was there for
        # legacy dev purposes
        cur_weights = cur_weights.reshape(cur_weights.shape[:4])
        # Convert theano weights to tensorflow
        if len(cur_weights.shape) == 4:
            cur_weights = cur_weights.transpose((2, 3, 1, 0))

        # For biases
        cur_biases = model[_old][_old + ".b"]

        # Add to feed_dict after
        feed_dict[placeholders[_new + "/weights"]] = cur_weights
        feed_dict[placeholders[_new + "/biases"]] = cur_biases

    return feed_dict


def load_legacy_network(supervisor, subtask, load_dir):
    """Load function for our old framework"""

    # Lazy loading to prevent import issues
    from utils import loadh5

    print("[{}] Checking if old pre-trained weights exists in {}"
          "".format(subtask, load_dir))
    model_file = os.path.join(load_dir, "model.h5")
    norm_file = os.path.join(load_dir, "norm.h5")
    base_file = os.path.join(load_dir, "base.h5")

    if os.path.exists(model_file) and os.path.exists(norm_file) and \
       os.path.exists(base_file):
        model = loadh5(model_file)
        norm = loadh5(norm_file)
        base = loadh5(base_file)
        # Load the input normalization parameters
        supervisor.network.mean["kp"] = float(norm["mean_x"])
        supervisor.network.mean["ori"] = float(norm["mean_x"])
        supervisor.network.mean["desc"] = float(base["patch-mean"])
        supervisor.network.std["kp"] = float(norm["std_x"])
        supervisor.network.std["ori"] = float(norm["std_x"])
        supervisor.network.std["desc"] = float(base["patch-std"])
        # Load weights for the component
        supervisor.network.legacy_load_func[subtask](supervisor.sess, model)
        print("[{}] Loaded previously trained weights".format(subtask))

        return True

    else:
        print("[{}] No pretrained weights from the old framework"
              "".format(subtask))

        return False


#
# legacy.py ends here
