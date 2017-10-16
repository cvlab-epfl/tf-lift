# network.py ---
#
# Filename: network.py
# Description:
# Author: Kwang Moo Yi
# Maintainer:
# Created: Thu Jul  6 15:38:49 2017 (+0200)
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


def make_theta(xyz, cs=None, scale=None, rr=0.5):
    """Make the theta to be used for the spatial transformer

    If cs is None, simply just do the translation only.

    """

    # get dx, dy, dz
    dx = xyz[:, 0]
    dy = xyz[:, 1]
    dz = xyz[:, 1]
    # compute the resize from the largest scale image
    reduce_ratio = rr
    dr = (reduce_ratio) * (2.0)**dz

    if cs is None:
        c = tf.ones_like(dx)
        s = tf.zeros_like(dx)
    else:
        c = cs[:, 0]
        s = cs[:, 1]

    theta = tf.stack(
        [dr * c, -dr * s, dx,
         dr * s,  dr * c, dy], axis=1)

    return theta


#
# network.py ends here
