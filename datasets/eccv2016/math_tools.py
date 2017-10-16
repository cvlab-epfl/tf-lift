# math_tools.py ---
#
# Filename: math_tools.py
# Description:
# Author: Kwang Moo Yi
# Maintainer:
# Created: Fri Feb 19 18:04:58 2016 (+0100)
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
import six


# ------------------------------------------------------------------------------
# Math functions
def softmax(val, axis, softmax_strength):
    ''' Soft max function used for cost function '''

    import theano
    import theano.tensor as T
    floatX = theano.config.floatX
    softmax_strength = np.cast[floatX](softmax_strength)

    if softmax_strength < 0:
        res_after_max = T.max(val, axis=axis)
    else:
        res_after_max = np.cast[floatX](1.0) / softmax_strength \
            * T.log(T.mean(T.exp(
                softmax_strength * (val - T.max(val,
                                                axis=axis,
                                                keepdims=True))
            ), axis=axis)) \
            + T.max(val, axis=axis)
    return res_after_max


def softargmax(val, axis, softargmax_strength):
    ''' Soft argmax function used for cost function '''

    import theano
    import theano.tensor as T
    floatX = theano.config.floatX

    # The implmentation only works for single axis
    assert isinstance(axis, int)

    if softargmax_strength < 0:
        res = T.argmax(val, axis=axis)
    else:
        safe_exp = T.exp(softargmax_strength * (
            val - T.max(val, axis=axis, keepdims=True)))
        prob = safe_exp / T.sum(safe_exp, axis=axis, keepdims=True)
        ind = T.arange(val.shape[axis], dtype=floatX)
        res = T.sum(prob * ind, axis=axis) / T.sum(prob, axis=axis)

    return res


def batch_inc_mean(data, batch_size, axis=None):
    """Computes standard deviation of the data using batches.

    Parameters
    ----------
    data: np.ndarray or h5py.Dataset
        Data to compute the standar deviation for.

    batch_size: int
        Size of the working batch.

    axis: None or tuple (optional)
        Axis in which we should take the standar deviation on

    Returns
    -------
    mean: np.ndarray or float
        Mean. ndarray if axis is not None, float otherwise.

    """

    if axis is not None:
        raise NotImplementedError('Working on arbitrary axis'
                                  ' is not yet implemented')

    else:
        numel = np.prod(data.shape)
        dim1_batch_size = int(np.floor(batch_size / np.prod(data.shape[1:])))
        num_batch = int(np.ceil(float(data.shape[0]) / float(dim1_batch_size)))

        mean = 0.0
        for idx_batch in six.moves.xrange(num_batch):
            idx_s = idx_batch * dim1_batch_size
            idx_e = min((idx_batch + 1) * dim1_batch_size, data.shape[0])
            x = data[idx_s:idx_e].flatten()
            mean += float(np.sum(x))

        mean = mean / float(numel)

    return mean


def batch_inc_std(data, mean, batch_size, axis=None):
    """Computes standard deviation of the data using batches.

    Parameters
    ----------
    data: np.ndarray or h5py.Dataset
        Data to compute the standar deviation for.

    mean: np.ndarray
        Precomputed mean of the data.

    batch_size: int
        Size of the working batch.

    axis: None or tuple (optional)
        Axis in which we should take the standar deviation on

    Returns
    -------
    std: np.ndarray or float
        Standar deviation. ndarray if axis is not None, float otherwise.

    """

    if axis is not None:
        raise NotImplementedError('Working on arbitrary axis'
                                  ' is not yet implemented')

    else:
        numel = np.prod(data.shape)
        dim1_batch_size = int(np.floor(batch_size / np.prod(data.shape[1:])))
        num_batch = int(np.ceil(float(data.shape[0]) / float(dim1_batch_size)))

        M2 = 0.0
        for idx_batch in six.moves.xrange(num_batch):
            idx_s = idx_batch * dim1_batch_size
            idx_e = min((idx_batch + 1) * dim1_batch_size, data.shape[0])
            x = data[idx_s:idx_e].flatten()
            M2 += float(np.sum((x - mean)**2))

        std = np.sqrt(M2 / float(numel - 1))

    return std


# ------------------------------------------------------------------------------
# Circle related functions
def getIntersectionOfCircles(x1, y1, r1, x2, y2, r2):

    # see http://mathworld.wolfram.com/Circle-CircleIntersection.html

    d = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

    # if d >= r1 + r2:
    #     return 0

    intersection = np.zeros_like(d)
    valid = np.where(d >= r1 + r2)[0]

    d = d[valid]
    r1 = r1[valid]
    r2 = r2[valid]

    intersection[valid] = (
        r2**2 * np.arccos((d**2 + r2**2 - r1**2) / (2.0 * d * r2)) +
        r1**2 * np.arccos((d**2 + r1**2 - r2**2) / (2.0 * d * r1)) -
        0.5 * np.sqrt(
            (-d + r2 + r1) * (d + r2 - r1) * (d - r2 + r1) * (d + r2 + r1)
        )
    )

    return intersection


def getInterNUnionOfCircles(x1, y1, r1, x2, y2, r2):

    intersection = getIntersectionOfCircles(x1, y1, r1, x2, y2, r2)
    union = np.pi * r1**2 + np.pi * r2**2 - intersection

    return intersection, union


# ------------------------------------------------------------------------------
# Rectangle related functions
def getIntersectionOfRectangles(x1, y1, r1, x2, y2, r2):
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
        # import theano
        import theano.tensor as T

        cm = T

    # intersection computation
    inter_w = cm.minimum(x1 + r1, x2 + r2) - cm.maximum(x1 - r1, x2 - r2)
    inter_w = cm.maximum(0, inter_w)
    inter_h = cm.minimum(y1 + r1, y2 + r2) - cm.maximum(y1 - r1, y2 - r2)
    inter_h = cm.maximum(0, inter_h)

    return inter_w * inter_h


def getIoUOfRectangles(x1, y1, r1, x2, y2, r2):
    """Computes Intersection over Union of Rectangles

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

    # compute intersection and union
    intersection = getIntersectionOfRectangles(x1, y1, r1, x2, y2, r2)
    union = (2.0 * r1)**2.0 + (2.0 * r2)**2.0 - intersection

    return intersection / union


# -----------------------------------------------------------------------------
# For normalization
def _subtractive_norm_make_coef(norm_kernel, input_hw):
    """Creates the coef matrix for accounting borders when applying
    subtractive normalization.

    Parameters
    ----------
    norm_kernel : np.ndarray, 2d
        Normalized kernel applied for subtractive normalization.

    input_hw : tuple
        Height and width of the input image (patch) for the
        SubtractiveNormalizationLayer.

    """

    assert np.isclose(np.sum(norm_kernel), 1.0)

    # This allows our mean computation to compensate for the border area,
    # where you have less terms adding up. Torch used convolution with a
    # ``one'' image, but since we do not want the library to depend on
    # other libraries with convolutions, we do it manually here.
    coef = np.ones(input_hw, dtype='float32')
    pad_x = norm_kernel.shape[1] // 2
    pad_y = norm_kernel.shape[0] // 2

    # Corners
    # for the top-left corner
    tl_cumsum_coef = np.cumsum(np.cumsum(
        norm_kernel[::-1, ::-1], axis=0), axis=1)[::1, ::1]
    coef[:pad_y + 1, :pad_x + 1] = tl_cumsum_coef[pad_y:, pad_x:]
    # for the top-right corner
    tr_cumsum_coef = np.cumsum(np.cumsum(
        norm_kernel[::-1, ::1], axis=0), axis=1)[::1, ::-1]
    coef[:pad_y + 1, -pad_x - 1:] = tr_cumsum_coef[pad_y:, :pad_x + 1]
    # for the bottom-left corner
    bl_cumsum_coef = np.cumsum(np.cumsum(
        norm_kernel[::1, ::-1], axis=0), axis=1)[::-1, ::1]
    coef[-pad_y - 1:, :pad_x + 1] = bl_cumsum_coef[:pad_y + 1, pad_x:]
    # for the bottom-right corner
    br_cumsum_coef = np.cumsum(np.cumsum(
        norm_kernel[::1, ::1], axis=0), axis=1)[::-1, ::-1]
    coef[-pad_y - 1:, -pad_x - 1:] = br_cumsum_coef[:pad_y + 1, :pad_x + 1]

    # Sides
    tb_slice = slice(pad_y + 1, -pad_y - 1)
    # for the left side
    fill_value = tl_cumsum_coef[-1, pad_x:]
    coef[tb_slice, :pad_x + 1] = fill_value.reshape([1, -1])
    # for the right side
    fill_value = br_cumsum_coef[0, :pad_x + 1]
    coef[tb_slice, -pad_x - 1:] = fill_value.reshape([1, -1])
    lr_slice = slice(pad_x + 1, -pad_x - 1)
    # for the top side
    fill_value = tl_cumsum_coef[pad_y:, -1]
    coef[:pad_y + 1, lr_slice] = fill_value.reshape([-1, 1])
    # for the right side
    fill_value = br_cumsum_coef[:pad_y + 1, 0]
    coef[-pad_y - 1:, lr_slice] = fill_value.reshape([-1, 1])

    # # code for validation of above
    # img = np.ones_like(input, dtype='float32')
    # import cv2
    # coef_cv2 = cv2.filter2D(img, -1, norm_kernel,
    #                         borderType=cv2.BORDER_CONSTANT)

    return coef

_lcn_make_coef = _subtractive_norm_make_coef


#
# math_tools.py ends here
