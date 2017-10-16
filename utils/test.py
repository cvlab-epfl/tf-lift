# test.py ---
#
# Filename: test.py
# Description:
# Author: Kwang Moo Yi
# Maintainer:
# Created: Thu Jul  6 16:24:22 2017 (+0200)
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

import cv2
import numpy as np
import tensorflow as tf

from six.moves import xrange


def compute_auc(x, y):
    """Compute AUCs given curve (x, y)

    Parameters
    ----------
    x : np.ndarray, float, 1d
        The values in x axis

    y : np.ndarray, float, 1d
        The values in y axis
    """
    # AUC for precision-recall
    dx = x[1:] - x[:-1]     # bin width
    y_avg = (y[1:] + y[:-1]) / 2.0  # avg y in bin
    # Below expression but without loops
    id_dx_valid = np.cast[float](dx > 0)
    auc = np.sum(y_avg * dx * id_dx_valid)
    # auc = 0
    # for _dx, _y_avg in zip(dx, y_avg):
    #     if _dx > 0:
    #         auc += _dx * _y_avg

    return auc


def eval_descs(d1, d2, d3, neg_weight):

    # Compute descriptor distances
    pair_dists = np.sqrt(np.sum((d1 - d2)**2, axis=1)).flatten()
    nonpair_dists = np.sqrt(np.sum((d1 - d3)**2, axis=1)).flatten()

    # Make dists and labels
    dists = np.concatenate([pair_dists, nonpair_dists], axis=0)
    labels = np.concatenate([
        np.ones_like(pair_dists), np.zeros_like(nonpair_dists)], axis=0)

    # create curves by hand
    num_th = 1000

    # LATER: Check if neg_per_pos is working

    # -------------------------------------------------
    # Do some smart trick to only sort and compare once
    # -------------------------------------------------
    # sort labels according to data indices
    sorted_labels = labels[np.argsort(dists)]
    # get thresholds using histogram function
    num_in_bin, thr = np.histogram(dists, num_th - 1)
    num_in_bin = np.concatenate([np.zeros((1,)), num_in_bin])
    # alocate
    prec = np.zeros((num_th,))
    rec = np.zeros((num_th,))
    tnr = np.zeros((num_th,))
    # for each bin
    idx_s = 0
    retrieved_pos = 0
    retrieved_neg = 0
    total_pos = np.sum(sorted_labels == 1).astype(float)
    total_neg = np.sum(sorted_labels == 0).astype(float) * neg_weight
    for i in xrange(num_th):
        # The number of elements we need to look at
        idx_e = int(idx_s + num_in_bin[i])
        # look at the labels to see how many are right/wrong and accumulate
        retrieved_pos += np.sum(sorted_labels[idx_s:idx_e] == 1).astype(float)
        retrieved_neg += np.sum(
            sorted_labels[idx_s:idx_e] == 0).astype(float) * neg_weight
        # move start point
        idx_s = idx_e

        retrieved = retrieved_pos + retrieved_neg
        if retrieved > 0:
            prec[i] = retrieved_pos / retrieved
        else:
            # precision is one, is this correct?
            prec[i] = 1.0
        rec[i] = retrieved_pos / total_pos
        tnr[i] = 1 - retrieved_neg / total_neg

    tpr = rec

    return compute_auc(rec, prec)


def eval_kps(xyzs, poss, scores, r_base):
    """Evaluate keypoints based on their locations and scores.


    Parameters
    ----------

    xyzs: the coordinates of the keypoints for P1, P2, P3, and P4

    poss: the ground-truth location of the SfM points for each patch

    scores: scores for each patch

    neg_weight: Weight of the negative sample

    r_base: float
        Base multiplier to apply to z to get rectangle support regions' half
        width. Typical computation would be:

        >>> r_base = (float(config.desc_input_size) /
                      float(get_patch_size(config))) / 6.0

    """

    # Let's convert everything to numpy arrays just to make our lives easier
    xyzs = np.asarray(xyzs)
    poss = np.asarray(poss)
    scores = np.asarray(scores)

    # Check for each keypoint if it is repeated. Note that we only compute the
    # overlap for *corresponding* pairs.
    #
    # th = 0.4
    # is_repeated = (get_kp_overlap(xyzs[0], poss[0],
    #                               xyzs[1], poss[1], r_base) >= 0.4)
    kp_overlap = get_kp_overlap(xyzs[0], poss[0], xyzs[1], poss[1],
                                r_base, mode='rectangle').flatten()

    # set number of thresholds we want to test
    num_th = 1000
    max_score = np.max(scores)
    min_score = np.min(scores)
    eps = 1e-10
    ths = np.linspace(max_score, min_score - 2 * eps, num_th) + eps

    # alocate
    prec = np.zeros((num_th,))
    rec = np.zeros((num_th,))
    tnr = np.zeros((num_th,))
    # Number of all positive pairs
    total_pos_pair = float(len(xyzs[0]))
    # Number of all non-SfM keypoints
    total_non_sfm_kp = float(len(xyzs[3]))
    for i in xrange(len(ths)):
        # set threshold
        th = ths[i]
        # find all that are selected as positives
        retv = [None] * 4
        for idx_branch in xrange(4):
            retv[idx_branch] = np.where(scores[idx_branch] >= th)[0]
        # number of retrieved keypoins. No balancing is needed since we don't
        # have repetitions any more
        retrieved_kp = sum([float(len(retv[_i])) for _i in xrange(4)])
        # Number of retrieved SfM keypoints. Again, no balancing
        retrieved_sfm_kp = sum([float(len(retv[_i])) for _i in xrange(3)])
        # number of retrieved non SfM keypoints
        retrieved_non_sfm_kp = float(len(retv[3]))
        # find positive pairs that are recovered (ones that are repeated)
        # check if both of the pairs are retrieved
        is_pair_retrieved = scores[1][retv[0]] >= th
        # get cumulative overlap
        retrieved_pos_pair = float(
            np.sum(is_pair_retrieved * kp_overlap[retv[0]]))

        if retrieved_kp > 0:
            prec[i] = retrieved_sfm_kp / retrieved_kp
        else:
            # precision is one, is this correct?
            prec[i] = 1.0
        rec[i] = retrieved_pos_pair / total_pos_pair
        tnr[i] = 1 - retrieved_non_sfm_kp / total_non_sfm_kp

        # if i == 500:
        #     import IPython
        #     IPython.embed()

    tpr = rec

    #, computeAUC(tnr[::-1], tpr[::-1]), prec, rec
    return compute_auc(rec, prec)


def rebase_xyz(xyz, base):
    """Recomputes 'xyz' with center being 'base'.

    Parameters
    ----------
    xyz: theano.tensor.matrix
       'xyz' to be modified to have center at 'base'.

    base: theano.tensor.matrix
        The new center position.

    Notes
    -----
    This function is required as xy is affected by which z you are having.
    """

    # Move to the same z.  Notice how we set the scaler here. For
    # example, if base == 1 then it means that we are moving to a
    # lower scale, so we need to multiply x and y by two.
    new_z = xyz[:, 2] - base[:, 2]
    scaler = 2.0**base[:, 2]
    new_x = xyz[:, 0] * scaler
    new_y = xyz[:, 1] * scaler

    # Move x and y. The new scaler here will be based on where x and y
    # are after moving, i.e. new_z. For example, if new_z == 1, then
    # we need to move x and y in half the value of the base x and y.
    scaler = 2.0**(-new_z)
    new_x -= base[:, 0] * scaler
    new_y -= base[:, 1] * scaler

    if isinstance(new_x, np.ndarray):
        new_xyz = np.concatenate([
            v.reshape([-1, 1]) for v in [new_x, new_y, new_z]
        ], axis=1)
    else:
        new_xyz = tf.concat([
            tf.reshape(v, [-1, 1]) for v in [new_x, new_y, new_z]
        ], axis=1)

    return new_xyz


def get_rect_inter(x1, y1, r1, x2, y2, r2):
    """Computes Intersection of Rectangles

    Parameters
    ----------
    x1: ndarray or tensor
        x coordinates of the first rectangle.

    y1: ndarray or tensor
        y coordinates of the first rectangle.

    r1: ndarray or tensor
        Half width of the first rectangle (just to be consistent with
        circle impl).

    x2: ndarray or tensor
        x coordinates of the second rectangle.

    y2: ndarray or tensor
        y coordinates of the second rectangle.

    r2: ndarray or tensor
        Half width of the second rectangle (just to be consistent with
        circle impl).
    """

    # use theano or numpy depending on the variable
    if isinstance(x1, np.ndarray):
        # Checking if ndarray will prevent us from loading theano
        # unnecessarily.
        cm = np
    else:
        # import tensorflow (Note that the code works for theano if you just
        # uncomment below)
        # import theano.tensor as T
        # cm = T
        import tensorflow as tf
        cm = tf

    # intersection computation
    inter_w = cm.minimum(x1 + r1, x2 + r2) - cm.maximum(x1 - r1, x2 - r2)
    inter_w = cm.maximum(0.0, inter_w)
    inter_h = cm.minimum(y1 + r1, y2 + r2) - cm.maximum(y1 - r1, y2 - r2)
    inter_h = cm.maximum(0.0, inter_h)

    return inter_w * inter_h


def get_kp_overlap(xyz1, pos1, xyz2, pos2, r_base, mode='circle'):
    """Determines if keypoint is repeated

    Parameters
    ----------
    xyz1: ndarray, float, 3d
        Coordinates and scale of the first keypoint.

    pos1: ndarray, float, 3d
        Coordinates and scale of the ground truth position of the first
        keypoint.

    xyz2: ndarray, float, 3d
        Coordinates and scale of the second keypoint.

    pos1: ndarray, float, 3d
        Coordinates and scale of the ground truth position of the second
        keypoint.

    r_base: float
        Base multiplier to apply to z to get rectangle support regions' half
        width. Typical computation would be:

        >>> r_base = (float(myNet.config.nDescInputSize) /
                      float(myNet.config.nPatchSize)) / 6.0

        We use the 1.0/6.0 of the support region to compute the overlap

    Note
    ----
    Unlike the cost function, here we use the intersection of circles to be
    maximum compatible with the other benchmarks. However, this can be easily
    changed. Also, r_base is computed to use the original keypoints scale,
    whereas the cost function uses the descriptor support region.

    """

    # Rebase the two xyzs so that they can be compared.
    xyz1 = rebase_xyz(xyz1, pos1)
    xyz2 = rebase_xyz(xyz2, pos2)

    # Retrieve the keypoint support  region based on r_base.
    x1 = xyz1[:, 0]
    y1 = xyz1[:, 1]
    r1 = r_base * 2.0**xyz1[:, 2]

    x2 = xyz2[:, 0]
    y2 = xyz2[:, 1]
    r2 = r_base * 2.0**xyz2[:, 2]

    if mode == 'circle':
        raise NotImplementedError(
            "LATER: after the release, when somebody wants to do it")
        # inter = get_circle_inter(x1, y1, r1, x2, y2, r2)
        # union = r1**2.0 + r2**2.0 - inter
    elif mode == 'rectangle':
        inter = get_rect_inter(x1, y1, r1, x2, y2, r2)
        union = (2.0 * r1)**2.0 + (2.0 * r2)**2.0 - inter
    else:
        raise ValueError('Unsupported mode ' + mode)

    return (inter / union).astype("float32").reshape([-1, 1])


def draw_XYZS_to_img(XYZS, image_color, out_file_name):
    """ Drawing functino for displaying """

    # draw onto the original image
    if cv2.__version__[0] == '3':
        linetype = cv2.LINE_AA
    else:
        linetype = cv2.CV_AA

    [cv2.circle(image_color, tuple(np.round(pos).astype(int)),
                np.round(rad * 6.0).astype(int), (0, 255, 0), 2,
                lineType=linetype)
     for pos, rad in zip(XYZS[:, :2], XYZS[:, 2])]

    cv2.imwrite(out_file_name, image_color)


#
# test.py ends here
