# lift_kp.py ---
#
# Filename: lift_kp.py
# Description: WRITEME
# Author: Kwang Moo Yi
# Maintainer: Kwang Moo Yi
# Created: Wed Jun 28 20:01:16 2017 (+0200)
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
import tensorflow as tf

from layers import conv_2d, ghh, batch_norm
from modules.bypass import bypass_kp
from utils import get_patch_size_no_aug, get_tensor_shape, \
    image_summary_nhwc, softmax


def process(inputs, bypass, name, skip, config, is_training):
    """WRITEME.

    LATER: Clean up

    inputs: input to the network
    bypass: gt to by used when trying to bypass
    name: name of the siamese branch
    skip: whether to apply the bypass information

    """

    # let's look at the inputs that get fed into this layer except when we are
    # looking at the whole image
    if name != "img":
        image_summary_nhwc(name + "-input", inputs)

    if skip:
        return bypass_kp(bypass)

    # we always expect a dictionary as return value to be more explicit
    res = {}

    # now abuse cur_in so that we can simply copy paste
    cur_in = inputs

    # lets apply batch normalization on the input - we did not normalize the
    # input range!
    # with tf.variable_scope("input-bn"):
    #     if config.use_input_batch_norm:
    #         cur_in = batch_norm(cur_in, training=is_training)

    with tf.variable_scope("conv-ghh-1"):
        nu = 1
        ns = 4
        nm = 4
        cur_in = conv_2d(
            cur_in, config.kp_filter_size, nu * ns * nm, 1, "VALID")
        # Disable batch norm: does not make sense for testing
        # as we run on the whole image rather than a collection of patches
        # if config.use_batch_norm:
        #     cur_in = batch_norm(cur_in, training=is_training)
        cur_in = ghh(cur_in, ns, nm)

    res["scoremap-uncut"] = cur_in

    # ---------------------------------------------------------------------
    # Check how much we need to cut
    kp_input_size = config.kp_input_size
    patch_size = get_patch_size_no_aug(config)
    desc_input_size = config.desc_input_size
    rf = float(kp_input_size) / float(patch_size)

    input_shape = get_tensor_shape(inputs)
    uncut_shape = get_tensor_shape(cur_in)
    req_boundary = np.ceil(rf * np.sqrt(2) * desc_input_size / 2.0).astype(int)
    cur_boundary = (input_shape[2] - uncut_shape[2]) // 2
    crop_size = req_boundary - cur_boundary

    # Stop building the network outputs if we are building for the full image
    if name == "img":
        return res

    # # Debug messages
    # resized_shape = get_tensor_shape(inputs)
    # print(' -- kp_info: output score map shape {}'.format(uncut_shape))
    # print(' -- kp_info: input size after resizing {}'.format(resized_shape[2]))
    # print(' -- kp_info: output score map size {}'.format(uncut_shape[2]))
    # print(' -- kp info: required boundary {}'.format(req_boundary))
    # print(' -- kp info: current boundary {}'.format(cur_boundary))
    # print(' -- kp_info: additional crop size {}'.format(crop_size))
    # print(' -- kp_info: additional crop size {}'.format(crop_size))
    # print(' -- kp_info: final cropped score map size {}'.format(
    #     uncut_shape[2] - 2 * crop_size))
    # print(' -- kp_info: movement ratio will be {}'.format((
    #     float(uncut_shape[2] - 2.0 * crop_size) /
    #     float(kp_input_size - 1))))

    # Crop center
    cur_in = cur_in[:, crop_size:-crop_size, crop_size:-crop_size, :]
    res["scoremap"] = cur_in

    # ---------------------------------------------------------------------
    # Mapping layer to x,y,z
    com_strength = config.kp_com_strength
    # eps = 1e-10
    scoremap_shape = get_tensor_shape(cur_in)

    od = len(scoremap_shape)
    # CoM to get the coordinates
    pos_array_x = tf.range(scoremap_shape[2], dtype=tf.float32)
    pos_array_y = tf.range(scoremap_shape[1], dtype=tf.float32)

    out = cur_in
    max_out = tf.reduce_max(
        out, axis=list(range(1, od)), keep_dims=True)
    o = tf.exp(com_strength * (out - max_out))  # + eps
    sum_o = tf.reduce_sum(
        o, axis=list(range(1, od)), keep_dims=True)
    x = tf.reduce_sum(
        o * tf.reshape(pos_array_x, [1, 1, -1, 1]),
        axis=list(range(1, od)), keep_dims=True
    ) / sum_o
    y = tf.reduce_sum(
        o * tf.reshape(pos_array_y, [1, -1, 1, 1]),
        axis=list(range(1, od)), keep_dims=True
    ) / sum_o

    # Remove the unecessary dimensions (i.e. flatten them)
    x = tf.reshape(x, (-1,))
    y = tf.reshape(y, (-1,))

    # --------------
    # Turn x, and y into range -1 to 1, where the patch size is
    # mapped to -1 and 1
    orig_patch_width = (
        scoremap_shape[2] + np.cast["float32"](req_boundary * 2.0))
    orig_patch_height = (
        scoremap_shape[1] + np.cast["float32"](req_boundary * 2.0))

    x = ((x + np.cast["float32"](req_boundary)) / np.cast["float32"](
        (orig_patch_width - 1.0) * 0.5) -
        np.cast["float32"](1.0))
    y = ((y + np.cast["float32"](req_boundary)) / np.cast["float32"](
        (orig_patch_height - 1.0) * 0.5) -
        np.cast["float32"](1.0))

    # --------------
    # No movement in z direction
    z = tf.zeros_like(x)

    res["xyz"] = tf.stack([x, y, z], axis=1)

    # ---------------------------------------------------------------------
    # Mapping layer to x,y,z
    res["score"] = softmax(
        res["scoremap"], axis=list(range(1, od)),
        softmax_strength = config.kp_scoremap_softmax_strength)

    return res

#
# lift_kp.py ends here
