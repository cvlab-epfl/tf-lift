# losses.py ---
#
# Filename: losses.py
# Description: WRITEME
# Author: Kwang Moo Yi
# Maintainer: Kwang Moo Yi
# Created: Wed Jun 28 20:06:43 2017 (+0200)
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

from utils import get_rect_inter, rebase_xyz


def loss_overlap(kp_pos1, gt_pos1, kp_pos2, gt_pos2, r_base,
                 alpha_non_overlap=1.0):
    """Loss from the overlap between keypoints detected in P1 and P2

    Note that due to the random perturbation, we need to compensate back the
    scale and the positions.

    Parameters
    ----------

    WRITEME

    """

    # Rebase the coordinates to the same base.
    xyz1 = rebase_xyz(kp_pos1, gt_pos1)
    xyz2 = rebase_xyz(kp_pos2, gt_pos2)

    # x,y and r for first patch
    scaler = 2.0**xyz1[:, 2]
    x1 = xyz1[:, 0] * scaler
    y1 = xyz1[:, 1] * scaler
    r1 = r_base * scaler

    # x,y and r for second patch
    scaler = 2.0**xyz2[:, 2]
    x2 = xyz2[:, 0] * scaler
    y2 = xyz2[:, 1] * scaler
    r2 = r_base * scaler

    # compute intersection and union
    intersection = get_rect_inter(x1, y1, r1, x2, y2, r2)
    union = (2.0 * r1)**2.0 + (2.0 * r2)**2.0 - intersection

    # add to cost (this has max value of 1)
    cost = 1.0 - intersection / union

    # compute the non-overlapping region cost
    dx = abs(x1 - x2)
    gap_x = tf.nn.relu(dx - (r1 + r2))
    dy = abs(y1 - y2)
    gap_y = tf.nn.relu(dy - (r1 + r2))

    # divide by the sqrt(union) to more-or-less be in same range as
    # above cost
    cost += alpha_non_overlap * (gap_x + gap_y) / (union**0.5)

    return cost


def loss_classification(s1, s2, s3, s4):
    """Loss from the classification

    Note s1, s2, and s3 are positive whereas s4 is negative. We therefore need
    to balance the cost that comes out of this. The original implementation for
    doing this is not identical to the new implementation, which may cause some
    minor differences.

    """

    # Get cost with l2 hinge loss
    cost_p = tf.add_n([
        tf.nn.relu(1.0 - s1), tf.nn.relu(1.0 - s2), tf.nn.relu(1.0 - s3)])
    cost_n = tf.nn.relu(1.0 + s4)
    # Make sure the elements sum to one. i.e. balance them.
    cost = cost_p / 6.0 + cost_n / 2.0

    return cost


def loss_desc_pair(d1, d2):
    """Loss using the euclidean distance

    """

    # return tf.norm(d1 - d2, ord="euclidean", axis=1)
    return tf.sqrt(tf.reduce_sum(tf.square(d1 - d2), axis=1))


def loss_desc_non_pair(d1, d3, margin, d2=None):
    """Loss using the euclidean distance and the margin

    """

    pair_dist_1_to_3 = tf.sqrt(tf.reduce_sum(tf.square(d1 - d3), axis=1))
    if d2 is not None:
        pair_dist_2_to_3 = tf.sqrt(tf.reduce_sum(tf.square(d2 - d3), axis=1))
        return tf.nn.relu(margin - tf.minimum(pair_dist_1_to_3, pair_dist_2_to_3))
    else:
        return tf.nn.relu(margin - pair_dist_1_to_3)


def loss_desc_triplet(d1, d2, d3, margin, squared_loss=False, mine_negative=False):
    """Triplet loss.

    """

    d_pos = tf.sqrt(tf.reduce_sum(tf.square(d1 - d2), axis=1))
    pair_dist_1_to_3 = tf.sqrt(tf.reduce_sum(tf.square(d1 - d3), axis=1))
    if mine_negative:
        pair_dist_2_to_3 = tf.sqrt(tf.reduce_sum(tf.square(d2 - d3), axis=1))
        d_neg = tf.minimum(pair_dist_1_to_3, pair_dist_2_to_3)
    else:
        d_neg = pair_dist_1_to_3

    if squared_loss:
        return tf.nn.relu(tf.square(d_pos) - tf.square(d_neg) + margin)
    else:
        return tf.nn.relu(d_pos - d_neg + margin)


#
# losses.py ends here
