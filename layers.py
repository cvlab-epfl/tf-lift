# layers.py ---
#
# Filename: layers.py
# Description: Special layers not included in tensorflow
# Author: Kwang Moo Yi
# Maintainer: Kwang Moo Yi
# Created: Thu Jun 29 12:23:35 2017 (+0200)
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
import tensorflow.contrib.layers as tcl

from utils import get_tensor_shape, get_W_b_conv2d, get_W_b_fc


def leaky_relu(x, alpha=0.2):
    return tf.maximum(tf.minimum(0.0, alpha * x), x)


def relu(x):
    return tf.nn.relu(x)


def batch_norm(x, training, data_format="NHWC"):
    # return tcl.batch_norm(x, center=True, scale=True, is_training=training, data_format=data_format)
    if data_format == "NHWC":
        axis = -1
    else:
        axis = 1
    return tf.layers.batch_normalization(x, training=training, trainable=False, axis=axis)


def norm_spatial_subtractive(inputs, sub_kernel, data_format="NHWC"):
    """Performs the spatial subtractive normalization

    Parameters
    ----------

    inputs: tensorflow 4D tensor, NHWC format

    input to the network

    sub_kernel: numpy.ndarray, 2D matrix

    the subtractive normalization kernel

    """

    raise NotImplementedError(
        "This function is buggy! don't use before extensive debugging!")

    # ----------
    # Normalize kernel.
    # Note that unlike Torch, we don't divide the kernel here. We divide
    # when it is fed to the convolution, since we use it to generate the
    # coefficient map.
    kernel = sub_kernel.astype("float32")
    norm_kernel = (kernel / np.sum(kernel))

    # ----------
    # Compute the adjustment coef.
    # This allows our mean computation to compensate for the border area,
    # where you have less terms adding up. Torch used convolution with a
    # ``one'' image, but since we do not want the library to depend on
    # other libraries with convolutions, we do it manually here.
    input_shape = get_tensor_shape(inputs)
    assert len(input_shape) == 4
    if data_format == "NHWC":
        coef = np.ones(input_shape[1:3], dtype="float32")
    else:
        coef = np.ones(input_shape[2:], dtype="float32")
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

    # ----------
    # Extract convolutional mean
    # Make filter a c01 filter by repeating. Note that we normalized above
    # with the number of repetitions we are going to do.
    if data_format == "NHWC":
        norm_kernel = np.tile(norm_kernel, [input_shape[-1], 1, 1])
    else:
        norm_kernel = np.tile(norm_kernel, [input_shape[1], 1, 1])
    # Re-normlize the kernel so that the sum is one.
    norm_kernel /= np.sum(norm_kernel)
    # add another axis in from to make oc01 filter, where o is the number
    # of output dimensions (in our case, 1!)
    norm_kernel = norm_kernel[np.newaxis, ...]
    # # To pad with zeros, half the size of the kernel (only for 01 dims)
    # border_mode = tuple(s // 2 for s in norm_kernel.shape[2:])
    # Convolve the mean filter. Results in shape of (batch_size,
    # input_shape[1], input_shape[2], 1).
    # For tensorflow, the kernel shape is 01co, which is different.... why?!
    conv_mean = tf.nn.conv2d(
        inputs,
        norm_kernel.astype("float32").transpose(2, 3, 1, 0),
        strides=[1, 1, 1, 1],
        padding="SAME",
        data_format=data_format,
    )

    # ----------
    # Adjust convolutional mean with precomputed coef
    # This is to prevent border values being too small.
    if data_format == "NHWC":
        coef = coef[None][..., None].astype("float32")
    else:
        coef = coef[None, None].astype("float32")
    adj_mean = conv_mean / coef
    # # Make second dimension broadcastable as we are going to
    # # subtract for all channels.
    # adj_mean = T.addbroadcast(adj_mean, 1)

    # ----------
    # Subtract mean
    sub_normalized = inputs - adj_mean

    # # line for debugging
    # test = theano.function(inputs=[input], outputs=[sub_normalized])

    return sub_normalized


def pool_l2(inputs, ksize, stride, padding, data_format="NHWC"):
    """L2 pooling, NHWC"""

    if data_format == "NHWC":
        ksizes = [1, ksize, ksize, 1]
        strides = [1, stride, stride, 1]
    else:
        ksizes = [1, 1, ksize, ksize]
        strides = [1, 1, stride, stride]

    scaler = tf.cast(ksize * ksize, tf.float32)

    return tf.sqrt(
        scaler *                # Multiply since we want to sum
        tf.nn.avg_pool(
            tf.square(inputs),
            ksize=ksizes,
            strides=strides,
            padding=padding,
            data_format=data_format,
        ))


def pool_max(inputs, ksize, stride, padding, data_format="NHWC"):
    """max pooling, NHWC"""

    if data_format == "NHWC":
        ksizes = [1, ksize, ksize, 1]
        strides = [1, stride, stride, 1]
    else:
        ksizes = [1, 1, ksize, ksize]
        strides = [1, 1, stride, stride]

    return tf.nn.max_pool(
        inputs,
        ksize=ksizes,
        strides=strides,
        padding=padding,
        data_format=data_format,
    )


def conv_2d(inputs, ksize, nchannel, stride, padding, data_format="NHWC"):
    """conv 2d, NHWC"""

    if data_format == "NHWC":
        fanin = get_tensor_shape(inputs)[-1]
        strides = [1, stride, stride, 1]
    else:
        fanin = get_tensor_shape(inputs)[1]
        strides = [1, 1, stride, stride]
    W, b = get_W_b_conv2d(ksize=ksize, fanin=fanin, fanout=nchannel)
    conv = tf.nn.conv2d(
        inputs, W, strides=strides,
        padding=padding, data_format=data_format)

    return tf.nn.bias_add(conv, b, data_format=data_format)


def fc(inputs, fanout):
    """fully connected, NC """

    inshp = get_tensor_shape(inputs)
    fanin = np.prod(inshp[1:])

    # Flatten input if needed
    if len(inshp) > 2:
        inputs = tf.reshape(inputs, (inshp[0], fanin))

    W, b = get_W_b_fc(fanin=fanin, fanout=fanout)
    mul = tf.matmul(inputs, W)

    return tf.nn.bias_add(mul, b)


def ghh(inputs, num_in_sum, num_in_max, data_format="NHWC"):
    """GHH layer

    LATER: Make it more efficient

    """

    # Assert NHWC
    assert data_format == "NHWC"

    # Check validity
    inshp = get_tensor_shape(inputs)
    num_channels = inshp[-1]
    pool_axis = len(inshp) - 1
    assert (num_channels % (num_in_sum * num_in_max)) == 0

    # Abuse cur_in
    cur_in = inputs

    # # Okay the maxpooling and avgpooling functions do not like weird
    # # pooling. Just reshape to avoid this issue.
    # inshp = get_tensor_shape(inputs)
    # numout = int(inshp[1] / (num_in_sum * num_in_max))
    # cur_in = tf.reshape(cur_in, [
    #     inshp[0], numout, num_in_sum, num_in_max, inshp[2], inshp[3]
    # ])

    # Reshaping does not work for undecided input sizes. use split instead
    cur_ins_to_max = tf.split(
        cur_in, num_channels // num_in_max, axis=pool_axis)

    # Do max and concat them back
    cur_in = tf.concat([
        tf.reduce_max(cur_ins, axis=pool_axis, keep_dims=True) for
        cur_ins in cur_ins_to_max
    ], axis=pool_axis)

    # Create delta
    delta = (1.0 - 2.0 * (np.arange(num_in_sum) % 2)).astype("float32")
    delta = tf.reshape(delta, [1] * (len(inshp) - 1) + [num_in_sum])

    # Again, split into multiple pieces
    cur_ins_to_sum = tf.split(
        cur_in, num_channels // (num_in_max * num_in_sum),
        axis=pool_axis)

    # Do delta multiplication, sum, and concat them back
    cur_in = tf.concat([
        tf.reduce_sum(cur_ins * delta, axis=pool_axis, keep_dims=True) for
        cur_ins in cur_ins_to_sum
    ], axis=pool_axis)

    return cur_in

#
# layers.py ends here
