# layer.py ---
#
# Filename: layer.py
# Description:
# Author: Kwang Moo Yi
# Maintainer:
# Created: Thu Jul  6 15:37:55 2017 (+0200)
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


def get_W_b_conv2d(ksize, fanin, fanout):
    W = tf.get_variable(
        name="weights",
        shape=[ksize, ksize, fanin, fanout],
        # initializer=tf.random_normal_initializer(2e-2),
        initializer=tf.truncated_normal_initializer(
            stddev=2.0 / (ksize * ksize * fanin)),
    )
    b = tf.get_variable(
        name="biases",
        shape=[fanout],
        initializer=tf.constant_initializer(0.0),
    )

    return W, b


def get_W_b_fc(fanin, fanout):
    W = tf.get_variable(
        name="weights",
        shape=[fanin, fanout],
        # initializer=tf.random_normal_initializer(2e-2),
        initializer=tf.truncated_normal_initializer(stddev=2.0 / fanin),
    )
    b = tf.get_variable(
        name="biases",
        shape=[fanout],
        initializer=tf.constant_initializer(0.0),
    )
    return W, b


def softmax(val, axis, softmax_strength):
    ''' Soft max function used for cost function '''

    softmax_strength = np.cast["float32"](softmax_strength)

    if softmax_strength < 0:
        res_after_max = tf.reduce_max(val, axis=axis)
    else:
        res_after_max = np.cast["float32"](1.0) / softmax_strength \
            * tf.log(tf.reduce_mean(tf.exp(
                softmax_strength * (
                    val - tf.reduce_max(val, axis=axis, keep_dims=True)
                )), axis=axis)) + tf.reduce_max(val, axis=axis)

    return res_after_max


#
# layer.py ends here
