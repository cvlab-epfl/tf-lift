# bypass.py ---
#
# Filename: bypass.py
# Description: WRITEME
# Author: Kwang Moo Yi
# Maintainer: Kwang Moo Yi
# Created: Thu Jun 29 14:13:15 2017 (+0200)
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


def bypass_kp(xyz):
    """Use GT information for keypoint detector

    Should return the xyz coordinates, as well as a dummy score.

    """

    # we always expect a dictionary as return value to be more explicit
    res = {}

    # Bypass for xy coordiantes
    res["xyz"] = xyz

    # Bypass for score (a dummy)
    res["score"] = xyz[:, 0]

    return res


def bypass_ori(angle):
    """Use GT information for orientation estimator

    Should return the cosine and the sine

    """

    # we always expect a dictionary as return value to be more explicit
    res = {}

    # Bypass for angle
    res["cs"] = tf.concat([tf.cos(angle), tf.sin(angle)], axis=1)

    return res

#
# bypass.py ends here
