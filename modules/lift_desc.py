# lift_desc.py ---
#
# Filename: lift_desc.py
# Description: WRITEME
# Author: Kwang Moo Yi
# Maintainer: Kwang Moo Yi
# Created: Wed Jun 28 20:02:58 2017 (+0200)
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

from layers import batch_norm, conv_2d, norm_spatial_subtractive, pool_l2
from utils import image_summary_nhwc, loadh5


def process(inputs, bypass, name, skip, config, is_training):
    """WRITEME

    inputs: input to the network
    bypass: gt to by used when trying to bypass
    name: name of the siamese branch
    skip: whether to apply the bypass information

    Note
    ----

    We don't have to worry about the reuse flag here, since it is already dealt
    with in the higher level. We just need to inherit it.

    """

    # We never skip descriptor
    assert skip is False

    # we always expect a dictionary as return value to be more explicit
    res = {}

    # let's look at the inputs that get fed into this layer
    image_summary_nhwc(name + "-input", inputs)

    # Import the lift_desc_sub_kernel.h5 to get the kernel file
    # script_dir = os.path.dirname(os.path.realpath(__file__))
    # sub_kernel = loadh5(script_dir + "/lift_desc_sub_kernel.h5")["kernel"]

    # activation
    if config.desc_activ == "tanh":
        activ = tf.nn.tanh
    elif config.desc_activ == "relu":
        activ = tf.nn.relu
    else:
        raise RuntimeError('Unknown activation type')

    # pooling
    def pool(cur_in, desc_pool, ksize):
        if desc_pool == "l2_pool":
            return pool_l2(cur_in, ksize, ksize, "VALID")
        elif desc_pool == "max_pool":
            return tf.nn.max_pool(cur_in, (1, ksize, ksize, 1), (1, ksize, ksize, 1), "VALID")
        elif desc_pool == "avg_pool":
            return tf.nn.avg_pool(cur_in, (1, ksize, ksize, 1), (1, ksize, ksize, 1), "VALID")
        else:
            raise RuntimeError('Unknown pooling type')

    # now abuse cur_in so that we can simply copy paste
    cur_in = inputs

    # lets apply batch normalization on the input - we did not normalize the
    # input range!
    with tf.variable_scope("input-bn"):
        if config.use_input_batch_norm:
            cur_in = batch_norm(cur_in, training=is_training)

    with tf.variable_scope("conv-act-pool-norm-1"):
        cur_in = conv_2d(cur_in, 7, 32, 1, "VALID")
        if config.use_batch_norm:
            cur_in = batch_norm(cur_in, training=is_training)
        cur_in = activ(cur_in)
        cur_in = pool(cur_in, config.desc_pool, 2)
        # if config.use_subtractive_norm:
        #     cur_in = norm_spatial_subtractive(cur_in, sub_kernel)

    with tf.variable_scope("conv-act-pool-norm-2"):
        cur_in = conv_2d(cur_in, 6, 64, 1, "VALID")
        if config.use_batch_norm:
            cur_in = batch_norm(cur_in, training=is_training)
        cur_in = activ(cur_in)
        cur_in = pool(cur_in, config.desc_pool, 3)
        # if config.use_subtractive_norm:
        #     cur_in = norm_spatial_subtractive(cur_in, sub_kernel)

    with tf.variable_scope("conv-act-pool-3"):
        cur_in = conv_2d(cur_in, 5, 128, 1, "VALID")
        if config.use_batch_norm:
            cur_in = batch_norm(cur_in, training=is_training)
        cur_in = activ(cur_in)
        cur_in = pool(cur_in, config.desc_pool, 4)

    res["desc"] = tf.reshape(cur_in, (-1, 128))

    return res

#
# lift_desc.py ends here
