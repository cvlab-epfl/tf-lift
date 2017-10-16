# lift_ori.py ---
#
# Filename: lift_ori.py
# Description: WRITEME
# Author: Kwang Moo Yi
# Maintainer: Kwang Moo Yi
# Created: Wed Jun 28 20:02:50 2017 (+0200)
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

import tensorflow as tf

from layers import batch_norm, conv_2d, fc, ghh, pool_max
from modules.bypass import bypass_ori
from utils import image_summary_nhwc


def process(inputs, bypass, name, skip, config, is_training):
    """WRITEME.

    inputs: input to the network
    bypass: gt to by used when trying to bypass
    name: name of the siamese branch
    skip: whether to apply the bypass information

    """

    # let's look at the inputs that get fed into this layer
    image_summary_nhwc(name + "-input", inputs)

    if skip:
        return bypass_ori(bypass)

    # we always expect a dictionary as return value to be more explicit
    res = {}

    # now abuse cur_in so that we can simply copy paste
    cur_in = inputs

    # lets apply batch normalization on the input - we did not normalize the
    # input range!
    with tf.variable_scope("input-bn"):
        if config.use_input_batch_norm:
            cur_in = batch_norm(cur_in, training=is_training)

    with tf.variable_scope("conv-act-pool-1"):
        cur_in = conv_2d(cur_in, 5, 10, 1, "VALID")
        if config.use_batch_norm:
            cur_in = batch_norm(cur_in, training=is_training)
        cur_in = tf.nn.relu(cur_in)
        cur_in = pool_max(cur_in, 2, 2, "VALID")

    with tf.variable_scope("conv-act-pool-2"):
        cur_in = conv_2d(cur_in, 5, 20, 1, "VALID")
        if config.use_batch_norm:
            cur_in = batch_norm(cur_in, training=is_training)
        cur_in = tf.nn.relu(cur_in)
        cur_in = pool_max(cur_in, 2, 2, "VALID")

    with tf.variable_scope("conv-act-pool-3"):
        cur_in = conv_2d(cur_in, 3, 50, 1, "VALID")
        if config.use_batch_norm:
            cur_in = batch_norm(cur_in, training=is_training)
        cur_in = tf.nn.relu(cur_in)
        cur_in = pool_max(cur_in, 2, 2, "VALID")
    # res["ori_out3"] = cur_in

    with tf.variable_scope("fc-ghh-drop-4"):
        nu = 100
        ns = 4
        nm = 4
        cur_in = fc(cur_in, nu * ns * nm)
        # cur_in = fc(cur_in, nu)
        if config.use_batch_norm:
            cur_in = batch_norm(cur_in, training=is_training)
        if config.ori_activation == 'ghh':
            cur_in = ghh(cur_in, ns, nm)
        elif config.ori_activation == 'tanh':
            cur_in = tf.nn.tanh(cur_in)
        else:
            raise RuntimeError("Bad orientation rectifier")
        # cur_in = tf.nn.relu(cur_in)
        if config.use_dropout_ori:
            raise RuntimeError('Dropout not working properly!')
            cur_in = tf.nn.dropout(
                cur_in,
                keep_prob=1.0 - (0.3 * tf.cast(is_training, tf.float32)),
            )
    # res["ori_out4"] = cur_in

    with tf.variable_scope("fc-ghh-5"):
        nu = 2
        ns = 4
        nm = 4
        cur_in = fc(cur_in, nu * ns * nm)
        # cur_in = fc(cur_in, nu)
        if config.use_batch_norm:
            cur_in = batch_norm(cur_in, training=is_training)
        if config.ori_activation == 'ghh':
            cur_in = ghh(cur_in, ns, nm)
        elif config.ori_activation == 'tanh':
            cur_in = tf.nn.tanh(cur_in)
        else:
            raise RuntimeError("Bad orientation rectifier")
        # cur_in = tf.nn.relu(cur_in)
    # res["ori_out5"] = cur_in

    # with tf.variable_scope("fc-ghh-6"):
    #     cur_in = fc(cur_in, nu)
    # res["ori_out6"] = cur_in

    with tf.variable_scope("cs-norm"):
        eps = 1e-10
        # First, normalize according to the maximum of the two
        cur_in_abs_max = tf.reduce_max(tf.abs(cur_in), axis=1, keep_dims=True)
        cur_in = cur_in / tf.maximum(eps, cur_in_abs_max)
        # Add an epsilon to avoid singularity
        eps = 1e-3
        cur_in += tf.to_float(cur_in >= 0) * eps - tf.to_float(cur_in < 0) * eps
        # Now make norm one without worrying about div by zero
        cur_in_norm = tf.sqrt(tf.reduce_sum(tf.square(
            cur_in), axis=1, keep_dims=True))
        cur_in /= cur_in_norm

    res["cs"] = tf.reshape(cur_in, (-1, 2))

    return res


#
# lift_ori.py ends here
