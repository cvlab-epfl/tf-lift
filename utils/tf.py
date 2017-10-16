# tf.py ---
#
# Filename: tf.py
# Description:
# Author: Kwang Moo Yi
# Maintainer:
# Created: Thu Jul  6 15:35:36 2017 (+0200)
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
import tensorflow.contrib.slim as slim


def show_all_variables():
    # Adapted from original code at
    # https://github.com/carpedm20/simulated-unsupervised-tensorflow
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)


def image_summary_nhwc(name, img, max_outputs=1):
    """Image summary function for NHWC format"""

    return tf.summary.image(name, img, max_outputs)


def image_summary_nchw(name, img, max_outputs=1):
    """Image summary function for NCHW format"""

    return tf.summary.image(
        name, tf.transpose(img, (0, 2, 3, 1)), max_outputs)


def get_tensor_shape(tensor):

    return [_s if _s is not None else -1 for
            _s in tensor.get_shape().as_list()]


#
# tf.py ends here
