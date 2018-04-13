# helper.py ---
#
# Filename: helper.py
# Description:
# Author: Kwang Moo Yi
# Maintainer:
# Created: Tue Feb 23 15:18:50 2016 (+0100)
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


from __future__ import print_function

import os

import cv2
import h5py  # for hdf5
import numpy as np
import scipy.io  # for matlab
import six
from time import time

# dump tools
from utils import loadh5, saveh5
from datasets.eccv2016.math_tools import getIoUOfRectangles
# from Utils.transformations import quaternion_from_matrix


# # custom types
# from Utils.custom_types import paramStruct, pathConfig

# def load_geom(geom_file, geom_type, scale_factor, flip_R=False):
#    if geom_type == "calibration":
#        # load geometry file
#        geom_dict = loadh5(geom_file)
#        # Check if principal point is at the center
#        K = geom_dict["K"]
#        assert(K[0, 2] < 1e-3 and K[1, 2] < 1e-3)
#        # Rescale calbration according to previous resizing
#        S = np.asarray([[scale_factor, 0, 0],
#                        [0, scale_factor, 0],
#                        [0, 0, 1]])
#        K = np.dot(S, K)
#        geom_dict["K"] = K
#        # Transpose Rotation Matrix if needed
#        if flip_R:
#            R = geom_dict["R"].T.copy()
#            geom_dict["R"] = R
#        # append things to list
#        geom_list = []
#        geom_info_name_list = ["K", "R", "T", "imsize"]
#        for geom_info_name in geom_info_name_list:
#            geom_list += [geom_dict[geom_info_name].flatten()]
#        # Finally do K_inv since inverting K is tricky with theano
#        geom_list += [np.linalg.inv(geom_dict["K"]).flatten()]
#        # Get the quaternion from Rotation matrices as well
#        q = quaternion_from_matrix(geom_dict["R"])
#        geom_list += [q.flatten()]
#        # Also add the inverse of the quaternion
#        q_inv = q.copy()
#        np.negative(q_inv[1:], q_inv[1:])
#        geom_list += [q_inv.flatten()]
#        # Add to list
#        geom = np.concatenate(geom_list)
#
#    elif geom_type == "homography":
#        H = np.loadtxt(geom_file)
#        geom = H.flatten()
#
#    return geom


def create_perturb(orig_pos, nPatchSize, nDescInputSize, fPerturbInfo):
    """Create perturbations in the xyz format we use in the keypoint component.

    The return value is in x,y,z where x and y are scaled coordinates that are
    between -1 and 1. In fact, the coordinates are scaled so that 1 = 0.5 *
    (nPatchSize-1) + patch center.

    In case of the z, it is in log2 scale, where zero corresponds to the
    original scale of the keypoint. For example, z of 1 would correspond to 2
    times the original scale of the keypoint.

    Parameters
    ----------
    orig_pos: ndarray of float
        Center position of the patch in coordinates.

    nPatchSize: int
        Patch size of the entire patch extraction region.

    nDescInputSize: int
        Patch size of the descriptor support region.

    fPerturbInfo: ndarray
        Amount of maximum perturbations we allow in xyz.

    Notes
    -----
    The perturbation has to ensure that the *ground truth* support region in
    included.
    """

    # Check provided perturb info so that it is guaranteed that it will
    # generate perturbations that holds the ground truth support region
    assert fPerturbInfo[0] <= 1.0 - float(nDescInputSize) / float(nPatchSize)
    assert fPerturbInfo[1] <= 1.0 - float(nDescInputSize) / float(nPatchSize)

    # Generate random perturbations
    perturb_xyz = ((2.0 * np.random.rand(orig_pos.shape[0], 3) - 1.0) *
                   fPerturbInfo.reshape([1, 3]))

    return perturb_xyz


def apply_perturb(orig_pos, perturb_xyz, maxRatioScale):
    """Apply the perturbation to ``a'' keypoint.

    The xyz perturbation is in format we use in the keypoint component. See
    'create_perturb' for details.

    Parameters
    ----------
    orig_pos: ndarray of float (ndim == 1, size 3)
        Original position of a *single* keypoint in pixel coordinates.

    perturb_xyz: ndarray of float (ndim == 1, size 3)
        The center position in xyz after the perturbation is applied.

    maxRatioScale: float
        The multiplier that get's multiplied to scale to get half-width of the
        region we are going to crop.
    """

    # assert that we are working with only one sample
    assert len(orig_pos.shape) == 1
    assert len(perturb_xyz.shape) == 1

    # get the new scale
    new_pos_s = orig_pos[2] * (2.0**(-perturb_xyz[2]))

    # get xy to pixels conversion
    xyz_to_pix = new_pos_s * maxRatioScale

    # Get the new x and y according to scale. Note that we multiply the
    # movement we need to take by 2.0**perturb_xyz since we are moving at a
    # different scale
    new_pos_x = orig_pos[0] - perturb_xyz[0] * 2.0**perturb_xyz[2] * xyz_to_pix
    new_pos_y = orig_pos[1] - perturb_xyz[1] * 2.0**perturb_xyz[2] * xyz_to_pix

    perturbed_pos = np.asarray([new_pos_x, new_pos_y, new_pos_s])

    return perturbed_pos


def get_crop_range(xx, yy, half_width):
    """Function for retrieving the crop coordinates"""

    xs = np.cast['int'](np.round(xx - half_width))
    xe = np.cast['int'](np.round(xx + half_width))
    ys = np.cast['int'](np.round(yy - half_width))
    ye = np.cast['int'](np.round(yy + half_width))

    return xs, xe, ys, ye


def crop_patch(img, cx, cy, clockwise_rot, resize_ratio, nPatchSize):
    """Crops the patch.

    Crops with center at cx, cy with patch size of half_width and resizes to
    nPatchSize x nPatchsize patch.

    Parameters
    ----------
    img: np.ndarray
        Input image to be cropped from.

    cx: float
        x coordinate of the patch.

    cy: float
        y coordinate of the patch.

    clockwise_rot: float (degrees)
        clockwise rotation to apply when extracting

    resize_ratio: float
        Ratio of the resize. For example, ratio of two will crop a 2 x
        nPatchSize region.

    nPatchSize: int
        Size of the returned patch.

    Notes
    -----

    The cv2.warpAffine behaves in similar way to the spatial transformers. The
    M matrix should move coordinates from the original image to the patch,
    i.e. inverse transformation.

    """

    # Below equation should give (nPatchSize-1)/2 when M x [cx, 0, 1]',
    # 0 when M x [cx - (nPatchSize-1)/2*resize_ratio, 0, 1]', and finally,
    # nPatchSize-1 when M x [cx + (nPatchSize-1)/2*resize_ratio, 0, 1]'.
    dx = (nPatchSize - 1.0) * 0.5 - cx / resize_ratio
    dy = (nPatchSize - 1.0) * 0.5 - cy / resize_ratio
    M = np.asarray([[1. / resize_ratio, 0.0, dx],
                    [0.0, 1. / resize_ratio, dy],
                    [0.0, 0.0, 1.0]])
    # move to zero base before rotation
    R_pre = np.asarray([[1.0, 0.0, -(nPatchSize - 1.0) * 0.5],
                        [0.0, 1.0, -(nPatchSize - 1.0) * 0.5],
                        [0.0, 0.0, 1.0]])
    # rotate
    theta = clockwise_rot / 180.0 * np.pi
    R_rot = np.asarray([[np.cos(theta), -np.sin(theta), 0.0],
                        [np.sin(theta), np.cos(theta), 0.0],
                        [0.0, 0.0, 1.0]])
    # move back to corner base
    R_post = np.asarray([[1.0, 0.0, (nPatchSize - 1.0) * 0.5],
                         [0.0, 1.0, (nPatchSize - 1.0) * 0.5],
                         [0.0, 0.0, 1.0]])
    # combine
    R = np.dot(R_post, np.dot(R_rot, R_pre))

    crop = cv2.warpAffine(img, np.dot(R, M)[:2, :], (nPatchSize, nPatchSize))

    return crop


def load_patches(img, kp_in, y_in, ID_in, angle_in, fRatioScale, fMaxScale,
                 nPatchSize, nDescInputSize, in_dim, bPerturb, fPerturbInfo,
                 bReturnCoords=False, nAugmentedRotations=1,
                 fAugmentRange=180.0, fAugmentCenterRandStrength=0.0,
                 sAugmentCenterRandMethod="uniform", nPatchSizeAug=None, is_test=False):
    '''Loads Patches given img and list of keypoints

    Parameters
    ----------

    img: input grayscale image (or color)

    kp_in: 2-D array of keypoints [nbkp, 3+], Note that data is in order of
        [x,y,scale,...]

    y_in: 1-D array of keypoint labels (or target values)

    ID_in: 1-D array of keypoint IDs

    fRatioScale: float
        The ratio which gets multiplied to obtain the crop radius. For example
        if fRatioScale is (48-1) /2 /2, and the scale is 2, the crop region
        will be of size 48x48. Note that -1 is necessary as the center pixel is
        inclusive. In our previous implementation regarding ECCV submission, -1
        was negelected. This however should not make a big difference.

    fMaxScale: float
        Since we are trying to make scale space learning possible, we give list
        of scales for training. This should contain the maximum value, so that
        we consider the largest scale when cropping.

    nPatchSize: int
        Size of the patch (big one, not to be confused with nPatchSizeKp).

    nDescInputSize: int
        Size of the inner patch (descriptor support region). Used for computing
        the bounds for purturbing the patch location.

    in_dim: int
        Number of dimensions of the input. For example grayscale would mean
        `in_dim == 1`.

    bPerturb: boolean
        Whether to perturb when extracting patches

    fPerturbInfo: np.array (float)
        How much perturbation (in relative scale) for x, y, scale

    bReturnCoord: boolean
        Return groundtruth coordinates. Should be set to True for new
        implementations. Default is False for backward compatibility.

    nAugmentedRotations: int
        Number of augmented rotations (equaly spaced) to be added to the
        dataset. The original implementation should be equal to having this
        number set to 1.

    fAugmentRange: float (degrees)
        The range of the augnmented degree. For example, 180 would mean the
        full range, 90 would be the half range.

    fAugmentCenterRandStrength: float (degrees)
        The strength of the random to be applied to the center angle
        perturbation.

    sAugmentCenterRandMethod: string
        Name of the center randomness method.

    nPatchSizeAug: int
        Size of the patch when augmenting rotations. This is used to create
        smaller perturbations to ensure we do not crop outside the patch.

    '''

    # get max possible scale ratio
    maxRatioScale = fRatioScale * fMaxScale

    # check validity of  nPreRotPatchSize
    assert nAugmentedRotations >= 1
    # # Since we apply perturbations, we need to be at least sqrt(2) larger than
    # # the desired when random augmentations are introduced
    # if nAugmentedRotations > 1 or fAugmentCenterRandStrength > 0:
    #     nInitPatchSize = np.round(np.sqrt(2.0) * nPatchSize).astype(int)
    # else:
    #     nInitPatchSize = nPatchSize

    if nPatchSizeAug is None:
        nPatchSizeAug = nPatchSize
    assert nPatchSize <= nPatchSizeAug

    # pre-allocate maximum possible array size for data
    x = np.zeros((kp_in.shape[0] * nAugmentedRotations, in_dim,
                  nPatchSizeAug, nPatchSizeAug), dtype='uint8')
    y = np.zeros((kp_in.shape[0] * nAugmentedRotations,), dtype='float32')
    ID = np.zeros((kp_in.shape[0] * nAugmentedRotations,), dtype='int')
    pos = np.zeros((kp_in.shape[0] * nAugmentedRotations, 3), dtype='float')
    angle = np.zeros((kp_in.shape[0] * nAugmentedRotations,), dtype='float32')
    coords = np.tile(np.zeros_like(kp_in), (nAugmentedRotations, 1))

    # create perturbations
    # Note: the purturbation still considers only the nPatchSize
    perturb_xyz = create_perturb(kp_in, nPatchSize,
                                 nDescInputSize, fPerturbInfo)

#    import pdb
#    pdb.set_trace()

    # delete perturbations for the negatives (look at kp[6])
    # perturb_xyz[kp_in[:, 6] == 0] = 0
    perturb_xyz[y_in == 0] = 0

    idxKeep = 0
    for idx in six.moves.xrange(kp_in.shape[0]):

        # current kp position
        cur_pos = apply_perturb(kp_in[idx], perturb_xyz[idx], maxRatioScale)
        cx = cur_pos[0]
        cy = cur_pos[1]
        cs = cur_pos[2]

        # retrieve the half width acording to scale
        max_hw = cs * maxRatioScale

        # get crop range considering bigger area
        xs, xe, ys, ye = get_crop_range(cx, cy, max_hw * np.sqrt(2.0))

        # boundary check with safety margin
        safety_margin = 1
        # if xs < 0 or xe >= img.shape[1] or ys < 0 or ys >= img.shape[0]:
        if (xs < safety_margin or xe >= img.shape[1] - safety_margin or
                ys < safety_margin or ys >= img.shape[0] - safety_margin):
            continue

        # create an initial center orientation
        center_rand = 0
        if sAugmentCenterRandMethod == "uniform":
            # Note that the below will give zero when
            # `fAugmentCenterRandStrength == 0`. This effectively disables the
            # random perturbation.
            center_rand = ((np.random.rand() * 2.0 - 1.0) *
                           fAugmentCenterRandStrength)
        else:
            raise NotImplementedError(
                "Unknown random method "
                "sAugmentCenterRandMethod = {}".format(
                    sAugmentCenterRandMethod
                )
            )

        # create an array of rotations to used
        rot_diff_list = np.arange(nAugmentedRotations).astype(float)
        # compute the step between subsequent rotations
        rot_step = 2.0 * fAugmentRange / float(nAugmentedRotations)
        rot_diff_list *= rot_step

        for rot_diff in rot_diff_list:

            # the rotation to be applied for this patch
            crot_deg = rot_diff + center_rand
            crot_rad = crot_deg * np.pi / 180.0

            # Crop using the crop function
            # crop = img[ys:ye, xs:xe]
            # x[idxKeep, 0, :, :] = crop_patch(
            #     img, cx, cy, crot_deg,
            #     max_hw / (float(nPatchSizeAug - 1) * 0.5),
            #     nPatchSizeAug)     # note the nInitPatchSize
            cur_patch = crop_patch(
                img, cx, cy, crot_deg,
                max_hw / (float(nPatchSizeAug - 1) * 0.5),
                nPatchSizeAug)
            if len(cur_patch.shape) == 2:
                #                pdb.set_trace()
                cur_patch = cur_patch[..., np.newaxis]

            x[idxKeep] = cur_patch.transpose(2, 0, 1)

            # crop = img[ys:ye, xs:xe]

            # update target value and id
            y[idxKeep] = y_in[idx]
            ID[idxKeep] = ID_in[idx]
            # add crot (in radians), note that we are between -2pi and 0 for
            # compatiblity
            # angle[idxKeep] = crot_rad
            if is_test:
                angle[idxKeep] = ((angle_in[idx] + crot_rad) % (2.0 * np.pi) -
                                  (2.0 * np.pi))
            else:
                angle[idxKeep] = crot_rad % (2.0 * np.pi) - (2.0 * np.pi)

            # Store the perturbation (center of the patch is 0,0,0)
            new_perturb_xyz = perturb_xyz[idx].copy()
            xx, yy, zz = new_perturb_xyz
            rx = np.cos(crot_rad) * xx - np.sin(crot_rad) * yy
            ry = np.sin(crot_rad) * xx + np.cos(crot_rad) * yy
            rz = zz
            pos[idxKeep] = np.asarray([rx, ry, rz])

            # store the original pixel coordinates
            new_kp_in = kp_in[idx].copy()
            new_kp_in[3] = ((new_kp_in[3] + crot_rad) % (2.0 * np.pi) -
                            (2.0 * np.pi))
            coords[idxKeep] = new_kp_in

            idxKeep += 1

    # Delete unassigned
    x = x[:idxKeep]
    y = y[:idxKeep]
    ID = ID[:idxKeep]
    pos = pos[:idxKeep]
    angle = angle[:idxKeep]
    coords = coords[:idxKeep]

    if not bReturnCoords:
        return x.astype("uint8"), y, ID.astype("int"), pos, angle
    else:
        return x.astype("uint8"), y, ID.astype("int"), pos, angle, coords


def random_mine_non_kp_with_2d_distance(img, pos_kp, scale_hist,
                                        scale_hist_c, param):
    """Randomly mines negatives with 2d distance.

    This function looks at raw pixel distances, without considering the
    scale. The threshold sould be given by the parameter. This function was our
    first atempt which was not really successful.

    Parameters
    ----------
    img: np.ndrray
        Input image to be mined

    pos_kp: np.ndarray
        Positive keypoint locations that we should avoid when mining. Note that
        this can be either SfM points, or all SIFT keypoints.

    scale_hist: np.ndarray
        Scale distribution to randomly sample scale from. Should be the scale
        histogram of positive keypoints.

    scale_hist_c: np.ndarray
        Bin center position of the scale histogram. Note that numpy histogram
        (by default) stores the cutoff values where we want the center values.

    param: paramStruct object
        The parameter object which is read from the configuration.

    """

    # read param
    neg_2_mine = param.dataset.nNegPerImg
    neg_dist_th = param.dataset.nNegDistTh
    if 'fNegOverlapTh' in param.patch.__dict__.keys():
        raise ValueError('fNegOverlapTh should not be defined '
                         'when using distance method!')

    all_neg_kp = None

    num_iter = 0
    while neg_2_mine > 0:
        # random sample positions
        neg_kp = np.random.rand(neg_2_mine * 10, 2)  # randomly get 10 times
        neg_kp *= np.array([img.shape[1] - 1 - 2.0 * neg_dist_th,
                            img.shape[0] - 1 - 2.0 * neg_dist_th],
                           dtype='float').reshape([1, 2])  # adjust to borders
        neg_kp += neg_dist_th

        # remove too close keypoints
        dists = neg_kp.reshape([-1, 2, 1]) - pos_kp[:, :2].reshape([1, 2, -1])
        idx_far_enough = (dists**2).sum(axis=1).min(axis=1) > neg_dist_th
        neg_kp = neg_kp[idx_far_enough]

        # random shuffle and take neg_2_mine
        idx_shuffle = np.random.permutation(len(neg_kp))
        neg_kp = neg_kp[idx_shuffle[:min(neg_2_mine, len(idx_shuffle))]]

        # concatenate random scale
        random_scales = np.random.choice(scale_hist_c, size=(len(neg_kp), 1),
                                         p=scale_hist)
        neg_kp = np.concatenate([neg_kp, random_scales], axis=1)

        # concatenate dummy values
        neg_kp = np.concatenate([neg_kp,
                                 np.zeros((len(neg_kp), 1)),
                                 -1.0 * np.ones((len(neg_kp), 1)),
                                 np.zeros((len(neg_kp), 1))], axis=1)

        # concatenate to the final list
        if all_neg_kp is None:
            all_neg_kp = neg_kp
        else:
            all_neg_kp = np.concatenate([all_neg_kp, neg_kp])

        # update neg_2_mine
        neg_2_mine = param.dataset.nNegPerImg - len(all_neg_kp)

        # update number of iterations
        num_iter += 1

        if num_iter > 100:
            raise RuntimeError('I am taking too much')

        neg_kp = all_neg_kp[:param.dataset.nNegPerImg]

    return neg_kp


def random_mine_non_kp_with_3d_blocking(img, pos_kp, scale_hist,
                                        scale_hist_c, param,
                                        neg_per_iter=None,
                                        max_iter=100):
    """Randomly mines negatives with 3d blocking.

    This function uses overlap and if the overlap / proposed region is larget
    than the given threshold, we don't use that area.

    Parameters
    ----------
    img: np.ndrray
        Input image to be mined

    pos_kp: np.ndarray
        Positive keypoint locations that we should avoid when mining. Note that
        this can be either SfM points, or all SIFT keypoints.

    scale_hist: np.ndarray
        Scale distribution to randomly sample scale from. Should be the scale
        histogram of positive keypoints.

    scale_hist_c: np.ndarray
        Bin center position of the scale histogram. Note that numpy histogram
        (by default) stores the cutoff values where we want the center values.

    param: paramStruct object
        The parameter object which is read from the configuration.

    neg_per_iter: int or None (optional)
        If not given or set to None, we will mine 10 times the desired amount
        at each iteration. This may cause negatives overlapping with each
        other. To make sure none of the negatives are overlapping, use 1 for
        this argument.

    max_iter: int (optional)
        Maximum iterations to the looping to try mining. When he does not find
        after this, he will simply stop mining and maybe print a message, but
        it will not necessarily crash. I did it this way since this function is
        intended to run in a sub-process, which the individual crashes do not
        really matter. And also, if you can't find enough negatives after that
        mining, maybe you are better off just mining no more.

    """

    # read param
    neg_2_mine = param.dataset.nNegPerImg
    neg_overlap_th = param.patch.fNegOverlapTh
    if 'nNegDistTh' in param.dataset.__dict__.keys():
        raise ValueError('nNegDistTh should not be defined '
                         'when using distance method!')
    # To stay inside the image boundaries
    maxRatioScale = param.patch.fRatioScale * param.patch.fMaxScale
    # The multiplier to get support region
    ratioDescSupport = (param.patch.fRatioScale *
                        float(param.model.nDescInputSize) /
                        float(param.patch.nPatchSize))

    all_neg_kp = None

    num_iter = 0
    t_start = time()
    while neg_2_mine > 0:

        if neg_per_iter is None:
            neg_per_iter = neg_2_mine * 10

        # randomly sample scales
        random_scales = np.random.choice(scale_hist_c, size=(neg_per_iter, 1),
                                         p=scale_hist)

        # random sample positions with scale in mind - this will ensure we are
        # in the boundary
        neg_kp = np.random.rand(neg_per_iter, 2)  # randomly get 10 times
        rescale_mtx = np.array([img.shape[1] - 1,
                                img.shape[0] - 1],
                               dtype='float').reshape([1, 2])
        rescale_mtx = (rescale_mtx -
                       2.0 * maxRatioScale * random_scales.reshape([-1, 1]))
        neg_kp *= rescale_mtx

        # concatenate scale
        neg_kp = np.concatenate([neg_kp, random_scales], axis=1)

        # compute overlaps
        x2 = neg_kp[:, 0]
        y2 = neg_kp[:, 1]
        r2 = neg_kp[:, 2] * ratioDescSupport
        # reshape x2, y2, r2 so that we do all the combinations
        x2.shape = [-1, 1]
        y2.shape = [-1, 1]
        r2.shape = [-1, 1]
        x1 = pos_kp[:, 0]
        y1 = pos_kp[:, 1]
        r1 = pos_kp[:, 2] * ratioDescSupport
        # reshape x1, y1, r1 so that we do all the combinations
        x1.shape = [1, -1]
        y1.shape = [1, -1]
        r1.shape = [1, -1]
        max_overlap_with_pos = np.max(
            getIoUOfRectangles(x1, y1, r1, x2, y2, r2),
            axis=1
        )
        bCheckNeg = False
        if all_neg_kp is not None:
            bCheckNeg = len(all_neg_kp) > 0
        if bCheckNeg:
            # overlap with previously mined negatives
            x1 = all_neg_kp[:, 0]
            y1 = all_neg_kp[:, 1]
            r1 = all_neg_kp[:, 2] * ratioDescSupport
            # reshape x1, y1, r1 so that we do all the combinations
            x1.shape = [1, -1]
            y1.shape = [1, -1]
            r1.shape = [1, -1]
            max_overlap_with_neg = np.max(
                getIoUOfRectangles(x1, y1, r1, x2, y2, r2),
                axis=1
            )
            # now get overlap in terms of ratio
            r2.shape = [-1]
            max_overlap = np.maximum(max_overlap_with_pos,
                                     max_overlap_with_neg)
        else:
            r2.shape = [-1]
            max_overlap = max_overlap_with_pos

        # remove too close keypoints
        idx_far_enough = max_overlap <= neg_overlap_th
        neg_kp = neg_kp[idx_far_enough]

        # random shuffle and take neg_2_mine
        idx_shuffle = np.random.permutation(len(neg_kp))
        neg_kp = neg_kp[idx_shuffle[:min(neg_2_mine, len(idx_shuffle))]]

        # concatenate dummy values
        neg_kp = np.concatenate([neg_kp,
                                 np.zeros((len(neg_kp), 1)),
                                 -1.0 * np.ones((len(neg_kp), 1)),
                                 np.zeros((len(neg_kp), 1))], axis=1)

        # concatenate to the final list
        if all_neg_kp is None:
            all_neg_kp = neg_kp
        else:
            all_neg_kp = np.concatenate([all_neg_kp, neg_kp])

        # update neg_2_mine
        neg_2_mine = param.dataset.nNegPerImg - len(all_neg_kp)

        # update number of iterations
        num_iter += 1

        # # if num_iter == 1 and len(all_neg_kp) == 0:
        # import pdb
        # pdb.set_trace()

        if num_iter > max_iter:
            print('\nRan {0:d} iterations, but could not mine {1:d} on this image [{2:.02f} s.]'
                  ''.format(num_iter, neg_2_mine, time() - t_start))
            break

        neg_kp = all_neg_kp[:param.dataset.nNegPerImg]

    return neg_kp


def get_list_of_img(train_data_dir, dump_data_dir, param, mode):

    # Check if split file exists
    split_prefix = train_data_dir + 'split-'
    split_prefix += str(param.dataset.nTrainPercent) + '-'
    split_prefix += str(param.dataset.nValidPercent) + '-'
    split_prefix += str(param.dataset.nTestPercent) + '-'

    # If it does not exist, create one
    if not os.path.exists(split_prefix + mode + '.txt'):

        # Read list of images
        list_png_file = []
        for files in os.listdir(train_data_dir):
            if files.endswith(".png"):
                list_png_file = list_png_file + [files]

        # Shuffle the image list
        if not os.path.exists(dump_data_dir + 'permute_png_idx.h5'):
            print(' -- ' + mode + ': '
                  'Creating new shuffle for reading images')
            permute_png_idx = np.random.permutation(len(list_png_file))
            to_save = {"saveval": permute_png_idx}
            saveh5(to_save,
                   dump_data_dir + 'permute_png_idx.h5')
            # dt.save(permute_png_idx,
            #         dump_data_dir + 'permute_png_idx.h5')
        else:
            print(' -- ' + mode + ': '
                  'Loading shuffle for reading images from '
                  '{}'.format(dump_data_dir))
            to_load = loadh5(dump_data_dir +
                             'permute_png_idx.h5')
            permute_png_idx = to_load["saveval"]
            # permute_png_idx = dt.load(dump_data_dir +
            #                           'permute_png_idx.h5')

        list_png_file = [list_png_file[permute_png_idx[idx]] for idx in
                         range(len(list_png_file))]

        # Write to file (all three)
        f_train = open(split_prefix + 'train.txt', 'w')
        f_valid = open(split_prefix + 'valid.txt', 'w')
        f_test = open(split_prefix + 'test.txt', 'w')

        train_end = int(float(param.dataset.nTrainPercent) / 100.0 *
                        len(list_png_file))
        valid_end = int(float(param.dataset.nTrainPercent +
                              param.dataset.nValidPercent) / 100.0 *
                        len(list_png_file))

        for idx_png in six.moves.xrange(len(list_png_file)):
            if idx_png > valid_end:
                print(list_png_file[idx_png], file=f_test)
            elif idx_png > train_end:
                print(list_png_file[idx_png], file=f_valid)
            else:
                print(list_png_file[idx_png], file=f_train)

        f_train.close()
        f_valid.close()
        f_test.close()

    # Read the list
    list_png_file = list(np.loadtxt(split_prefix + mode + '.txt', dtype='str'))

    return list_png_file


def get_scale_hist(train_data_dir, param):

    # Check if scale histogram file exists
    hist_file_name = train_data_dir + 'scales-histogram-minsc-' \
        + str(param.dataset.fMinKpSize) + '.h5'
    if not os.path.exists(hist_file_name):

        # read all positive keypoint scales
        list_png_file = []
        for files in os.listdir(train_data_dir):
            if files.endswith(".png"):
                list_png_file = list_png_file + [files]
        all_scales = []
        for png_file in list_png_file:
            kp_file_name = train_data_dir + png_file.replace('.png', '_P.mat')
            cur_pos_kp = scipy.io.loadmat(kp_file_name)['TFeatures']
            cur_pos_kp = np.asarray(cur_pos_kp, dtype='float')
            all_scales += [cur_pos_kp[5, :].flatten()]
        all_scales = np.concatenate(all_scales)

        # make histogram
        hist, bin_edges = np.histogram(all_scales, bins=100)
        hist_c = (bin_edges[1:] + bin_edges[:-1]) * 0.5

        # save to h5 file
        with h5py.File(hist_file_name, 'w') as hist_file:
            hist_file['all_scales'] = all_scales
            hist_file['histogram_bins'] = hist
            hist_file['histogram_centers'] = hist_c

    # Load from the histogram file
    with h5py.File(hist_file_name, 'r') as hist_file:
        scale_hist = np.asarray(hist_file['histogram_bins'],
                                dtype=float).flatten()
        scale_hist /= np.sum(scale_hist)
        scale_hist_c = np.asarray(hist_file['histogram_centers']).flatten()

    return scale_hist, scale_hist_c

#
# helper.py ends here
