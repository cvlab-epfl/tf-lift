# lift.py ---
#
# Filename: lift.py
# Description: WRITEME
# Author: Kwang Moo Yi, Lin Chen
# Maintainer: Kwang Moo Yi
# Created: ?????
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

# Code:

import importlib

import numpy as np
import tensorflow as tf
from time import time
from tqdm import tqdm

from losses import (loss_classification, loss_desc_non_pair, loss_desc_pair,
                    loss_desc_triplet, loss_overlap)
from modules.spatial_transformer import transformer as transformer
from six.moves import xrange
from utils import (eval_descs, eval_kps, get_patch_size, get_patch_size_no_aug,
                   get_tensor_shape, image_summary_nhwc, make_theta,
                   show_all_variables)
# from utils.legacy import build_legacy


class Network(object):
    """The LIFT Network

    LATER
    - Do proper mean/std normalization?

    """

    def __init__(self, sess, config, dataset, force_mean_std=None):
        # Save pointer to the tensorflow session
        self.sess = sess
        # Save pointer to config
        self.config = config
        # Save pointer to the data module
        self.dataset = dataset
        # # Summaries to compute for this network
        # self.summary = []

        # Normalizer for the input data (they are raw images)
        # Currently normalized to be between -1 and 1
        self.mean = {}
        self.std = {}
        # Load values if they already exist
        if force_mean_std is not None:
            self.mean = force_mean_std["mean"]
            self.std = force_mean_std["std"]
        elif self.config.mean_std_type == "hardcoded":
            print("-- Using default values for mean/std")
            for _module in ["kp", "ori", "desc"]:
                self.mean[_module] = 128.0
                self.std[_module] = 128.0
        elif self.config.mean_std_type == "old":
            print("-- Using old (piccadilly) values for mean/std")
            self.mean["kp"] = 116.4368117568544249706974369473755359649658203125
            self.std["kp"] = 88.083076379771597430590190924704074859619140625
            self.mean["ori"] = 116.4368117568544249706974369473755359649658203125
            self.std["ori"] = 88.083076379771597430590190924704074859619140625
            self.mean["desc"] = 110.75389862060546875
            self.std["desc"] = 61.53688812255859375
        elif self.config.mean_std_type == "dataset":
            t = time()
            print("-- Recomputing dataset mean/std...")
            # Account for augmented sets
            if self.config.use_augmented_set:
                b = int((get_patch_size(config) -
                         get_patch_size_no_aug(config)) / 2)
            else:
                b = 0

            if b > 0:
                _d = self.dataset.data["train"]["patch"][:, :, b:-b, b:-b]
            else:
                _d = self.dataset.data["train"]["patch"][:, :, :, :]

            # Do this incrementally to avoid memory problems
            jump = 1000
            data_mean = np.zeros(_d.shape[0])
            data_std = np.zeros(_d.shape[0])
            for i in tqdm(range(0, _d.shape[0], jump)):
                data_mean[i:i + jump] = _d[i:i + jump].mean()
                data_std[i:i + jump] = _d[i:i + jump].std()
            data_mean = data_mean.mean()
            data_std = data_std.mean()
            print('-- Dataset mean: {0:.03f}, std = {1:.03f}'.format(data_mean, data_std))

            for _module in ["kp", "ori", "desc"]:
                self.mean[_module] = data_mean
                self.std[_module] = data_std
            print("-- Done in {0:.02f} sec".format(time() - t))
        elif self.config.mean_std_type == "batch":
            t = time()
            print("-- Will recompute mean/std per batch...")
        elif self.config.mean_std_type == "sample":
            t = time()
            print("-- Will recompute mean/std per sample...")
        elif self.config.mean_std_type == "sequence":
            t = time()
            print("-- Will recompute mean/std per sequence...")
            raise RuntimeError("TODO")
        else:
            raise RuntimeError("Unknown mean-std strategy")

        # Account for the keypoint scale change while augmenting rotations
        self.scale_aug = float(get_patch_size(self.config)) / \
            float(get_patch_size_no_aug(self.config))

        # Allocate placeholders
        with tf.variable_scope("placeholders"):
            self._build_placeholders()
        # Build the network
        with tf.variable_scope("network"):
            self._build_network()
        # Build loss
        with tf.variable_scope("loss"):
            self._build_loss()
        # Build the optimization op
        with tf.variable_scope("optimization"):
            self._build_optim()

        # Build the legacy component. This is only used for accessing old
        # framework weights. You can safely ignore this part
        # build_legacy(self)

        # Show all variables in the network
        show_all_variables()

        # Add all variables into histogram summary
        for _module in ["kp", "ori", "desc"]:
            for _param in self.params[_module]:
                tf.summary.histogram(_param.name, _param)

        # Collect all summary (Lazy...)
        self.summary = tf.summary.merge_all()

    def forward(self, subtask, cur_data):
        """Forward pass of the network.

        Runs the network and returns the losses for the subtask.

        """

        # parse cur_data and create a feed_dict
        feed_dict = self._get_feed_dict(subtask, cur_data)
        # set to train mode
        feed_dict[self.is_training] = True

        # import IPython
        # IPython.embed()

        # specify which loss we are going to retrieve
        fetch = {
            "loss": self.loss[subtask],
        }

        # run on the tensorflow session
        res = self.sess.run(fetch, feed_dict=feed_dict)

        # return the losses
        return res["loss"]

    def backward(self, subtask, cur_data, provide_summary=False):
        """Backward pass of the network.

        Runs the network and updates the parameters, as well as returning the
        summary if asked to do so.

        """

        # parse cur_data and create a feed_dict
        feed_dict = self._get_feed_dict(subtask, cur_data)
        # set to train mode
        feed_dict[self.is_training] = True

        # specify which optim we run depending on the subtask
        fetch = {
            "optim": self.optim[subtask],
        }
        if provide_summary:
            fetch["summary"] = self.summary

        # run on the tensorflow session
        try:
            res = self.sess.run(fetch, feed_dict=feed_dict)
        except:
            print("backward pass had an error, this iteration did nothing.")
            return None

        # return summary
        if provide_summary:
            return res["summary"]
        else:
            return None

    def validate(self, subtask, cur_data):
        """Validation process.

        Runs the network for the given data and returns validation results. The
        return value should be a single scalar, which lower is better.

        """

        # Get feed_dict for this batch
        feed_dict = self._get_feed_dict(subtask, cur_data)
        # set to train mode
        feed_dict[self.is_training] = False

        # Obtain xyz, scores, descs, whatever you need
        fetch = {}
        if subtask == "kp":
            # Obtain xyz's and scores
            for _i in xrange(1, 5):
                # Detector output
                fetch["xyz{}".format(_i)] \
                    = self.outputs["kp"]["P{}".format(_i)]["xyz"]
                # Warp ground truth if necessary
                aug_rot = self.inputs["aug_rot"] \
                    if self.config.augment_rotations else None
                fetch["pos{}".format(_i)] = self.transform_xyz(
                    self.inputs["xyz"],
                    aug_rot,
                    self.config.batch_size,
                    self.scale_aug,
                    transpose=True,
                    names=["P{}".format(_i)])
                # fetch["pos{}".format(_i)] \
                #     = self.inputs["xyz"]["P{}".format(_i)]
                fetch["score{}".format(_i)] \
                    = self.outputs["kp"]["P{}".format(_i)]["score"]
            res = self.sess.run(fetch, feed_dict=feed_dict)

            # Compute the AUC
            # neg_weight = self.config.neg_per_pos
            prauc = eval_kps(
                [res["xyz{}".format(_i + 1)] for _i in xrange(4)],
                [res["pos{}".format(_i + 1)]['P{}'.format(_i + 1)]
                    for _i in xrange(4)],
                [res["score{}".format(_i + 1)] for _i in xrange(4)],
                self.r_base,
            )
            # prauc = eval_kps(
            #     [res["xyz{}".format(_i + 1)] for _i in xrange(4)],
            #     [res["pos{}".format(_i + 1)] * self.scale_aug
            #         for _i in xrange(4)],
            #     [res["score{}".format(_i + 1)] for _i in xrange(4)],
            #     self.r_base,
            # )

            return 1.0 - prauc

        else:
            # Obtain descriptors for each patch
            for _i in xrange(1, 4):
                fetch["d{}".format(_i)] \
                    = self.outputs["desc"]["P{}".format(_i)]["desc"]
            res = self.sess.run(fetch, feed_dict=feed_dict)

            # Compute the AUC
            neg_weight = self.config.neg_per_pos
            prauc = eval_descs(
                res["d1"], res["d2"], res["d3"], neg_weight)

            return 1.0 - prauc

    def test(self, subtask, cur_data):
        """The test function to retrieve results for a single patch.

        LATER: Comment

        """

        if subtask == "kp":
            feed_dict = {
                self.inputs["img"]["img"]: cur_data,
                self.is_training: False,
            }
            fetch = {
                "scoremap": self.outputs["kp"]["img"]["scoremap-uncut"]
            }
            res = self.sess.run(fetch, feed_dict=feed_dict)
            return res["scoremap"]
        elif subtask == "ori":
            feed_dict = {
                self.inputs["patch"]["P1"]: cur_data["patch"],
                self.inputs["xyz"]["P1"]: cur_data["xyz"],
                self.is_training: False,
            }
            fetch = {
                "cs": self.outputs["ori"]["P1"]["cs"]
            }
            res = self.sess.run(fetch, feed_dict=feed_dict)
            return np.arctan2(res["cs"][:, 1], res["cs"][:, 0])
        elif subtask == "desc":
            feed_dict = {
                self.inputs["patch"]["P1"]: cur_data["patch"],
                self.inputs["xyz"]["P1"]: cur_data["xyz"],
                self.inputs["angle"]["P1"]: cur_data["angle"],
                self.is_training: False,
            }
            fetch = {
                "desc": self.outputs["desc"]["P1"]["desc"]
            }
            res = self.sess.run(fetch, feed_dict=feed_dict)
            return res["desc"]

    def _build_placeholders(self):
        """Builds Tensorflow Placeholders"""

        # The inputs placeholder dictionary
        self.inputs = {}
        # multiple types
        # LATER: label might not be necessary
        types = ["patch", "xyz", "angle"]
        if self.config.use_augmented_set:
            types += ["aug_rot"]
        for _type in types:
            self.inputs[_type] = {}

        # We *ARE* going to specify the input size, since the spatial
        # transformer implementation *REQUIRES* us to do so. Note that this
        # has to be dealt with in the validate loop.

        # batch_size = self.config.batch_size

        # Use variable batch size
        batch_size = None

        # We also read nchannel from the configuration. Make sure that the data
        # module is behaving accordingly
        nchannel = self.config.nchannel

        # Get the input patch size from config
        patch_size = float(get_patch_size(self.config))

        # Compute the r_base (i.e. overlap radius when computing the keypoint
        # overlaps.
        self.r_base = (float(self.config.desc_input_size) /
                       float(get_patch_size_no_aug(self.config)))

        # P1, P2, P3, P4 in the paper. P1, P2, P3 are keypoints, P1, P2
        # correspond, P1, and P3 don't correspond, P4 is a non-keypoint patch.
        for _name in ["P1", "P2", "P3", "P4"]:
            self.inputs["patch"][_name] = tf.placeholder(
                tf.float32, shape=[
                    batch_size, patch_size, patch_size, nchannel
                ], name=_name,
            )
            self.inputs["xyz"][_name] = tf.placeholder(
                tf.float32, shape=[
                    batch_size, 3,
                ], name=_name,
            )
            self.inputs["angle"][_name] = tf.placeholder(
                tf.float32, shape=[
                    batch_size, 1,
                ], name=_name,
            )
            if self.config.use_augmented_set:
                self.inputs["aug_rot"][_name] = {
                    "cs": tf.placeholder(
                        tf.float32, shape=[
                            batch_size, 2,
                        ], name=_name,
                    ),
                    "angle": tf.placeholder(
                        tf.float32, shape=[
                            batch_size, 1,
                        ], name=_name,
                    )}
            # Add to summary to view them
            image_summary_nhwc(
                "input/" + _name,
                self.inputs["patch"][_name],
            )

        # For Image based test
        self.inputs["img"] = {"img": tf.placeholder(
            tf.float32, shape=[
                1, None, None, nchannel
            ], name="img",
        )}

        # For runmode in dropout and batch_norm
        self.is_training = tf.placeholder(
            tf.bool, shape=(),
            name="is_training",
        )

    def _build_network(self):
        """Define all the architecture here. Use the modules if necessary."""

        # Import modules according to the configurations
        self.modules = {}
        for _key in ["kp", "ori", "desc"]:
            self.modules[_key] = importlib.import_module(
                "modules.{}".format(
                    getattr(self.config, "module_" + _key)))

        # prepare dictionary for the output and parameters of each module
        self.outputs = {}
        self.params = {}
        self.allparams = {}
        for _key in self.modules:
            self.outputs[_key] = {}
            self.params[_key] = []
            self.allparams[_key] = []
        # create a joint params list
        # NOTE: params is a list, not a dict!
        self.params["joint"] = []
        self.allparams["joint"] = []
        # create outputs placeholder for crop and rot
        self.outputs["resize"] = {}
        self.outputs["crop"] = {}
        self.outputs["rot"] = {}

        # Actual Network definition
        with tf.variable_scope("lift"):
            # Graph construction depends on the subtask
            subtask = self.config.subtask

            # ----------------------------------------
            # Initial resize for the keypoint module
            # Includes rotation when augmentations are used
            #
            if self.config.use_augmented_set:
                rot = self.inputs["aug_rot"]
            else:
                rot = None
            self._build_st(
                module="resize",
                xyz=None,
                cs=rot,
                names=["P1", "P2", "P3", "P4"],
                out_size=self.config.kp_input_size,
                reduce_ratio=float(get_patch_size_no_aug(self.config)) /
                float(get_patch_size(self.config)),
            )

            # ----------------------------------------
            # Keypoint Detector
            #
            # The keypoint detector takes each patch input and outputs (1)
            # "score": the score of the patch, (2) "xy": keypoint position in
            # side the patch. The score output is the soft-maximum (not a
            # softmax) of the scores. The position output from the network
            # should be in the form friendly to the spatial
            # transformer. Outputs are always dictionaries.
            # Rotate ground truth coordinates when augmenting rotations.
            aug_rot = self.inputs["aug_rot"] \
                if self.config.augment_rotations else None
            xyz_gt_scaled = self.transform_xyz(
                self.inputs["xyz"],
                aug_rot,
                self.config.batch_size,
                self.scale_aug,
                transpose=True,
                names=["P1", "P2", "P3", "P4"])
            self._build_module(
                module="kp",
                inputs=self.outputs["resize"],
                bypass=xyz_gt_scaled,
                names=["P1", "P2", "P3", "P4"],
                skip=subtask == "ori" or subtask == "desc",
            )

            # For image based test
            self._build_module(
                module="kp",
                inputs=self.inputs["img"],
                bypass=self.inputs["img"],  # This is a dummy
                names=["img"],
                skip=subtask != "kp",
                reuse=True,
                test_only=True,
            )

            # ----------------------------------------
            # The Crop Spatial Transformer
            # Output: use the same support region as for the descriptor
            #
            xyz_kp_scaled = self.transform_kp(
                self.outputs["kp"],
                aug_rot,
                self.config.batch_size,
                1 / self.scale_aug,
                transpose=False,
                names=["P1", "P2", "P3"])
            self._build_st(
                module="crop",
                xyz=xyz_kp_scaled,
                cs=aug_rot,
                names=["P1", "P2", "P3"],
                out_size=self.config.ori_input_size,
                reduce_ratio=float(self.config.desc_input_size) /
                float(get_patch_size(self.config)),
            )

            # ----------------------------------------
            # Orientation Estimator
            #
            # The orientation estimator takes the crop outputs as input and
            # outputs orientations for the spatial transformer to
            # use. Actually, since we output cos and sin, we can simply use the
            # *UNNORMALIZED* version of the two, normalize them, and directly
            # use it for our affine transform. In short it returns "cs": the
            # cos and the sin, but unnormalized. Outputs are always
            # dictionaries.
            # Bypass: just the GT angle
            if self.config.augment_rotations:
                rot = {}
                for name in ["P1", "P2", "P3"]:
                    rot[name] = self.inputs["angle"][name] - \
                        self.inputs["aug_rot"][name]["angle"]
            else:
                rot = self.inputs["angle"]
            self._build_module(
                module="ori",
                inputs=self.outputs["crop"],
                bypass=rot,
                names=["P1", "P2", "P3"],
                skip=subtask == "kp" or subtask == "desc",
            )

            # ----------------------------------------
            # The Rot Spatial Transformer.
            # - No rotation augmentation: 
            # Operates over the original patch with the ground truth angle when
            # bypassing. Otherwise, we combine the augmented angle and the
            # output of the orientation module.
            # We do not consider rotation augmentations for the descriptor.
            if self.config.augment_rotations:
                rot = self.chain_cs(
                    self.inputs["aug_rot"],
                    self.outputs["ori"],
                    names=["P1", "P2", "P3"])
                # rot = self.outputs["ori"]
                # xyz_desc_scaled = self.transform_kp(
                #     self.outputs["kp"],
                #     rot,
                #     self.config.batch_size,
                #     1 / self.scale_aug,
                #     transpose=False,
                #     names=["P1", "P2", "P3"])
            # elif self.config.use_augmented_set:
            else:
                rot = self.outputs["ori"]
                # xyz_desc_scaled = self.transform_kp(
                #     self.outputs["kp"],
                #     rot,
                #     self.config.batch_size,
                #     1 / self.scale_aug,
                #     transpose=False,
                #     names=["P1", "P2", "P3"])
            # else:
            #     rot = None
                # xyz_desc_scaled = self.inputs["xyz"]
            self._build_st(
                module="rot",
                xyz=xyz_kp_scaled,
                cs=rot,
                names=["P1", "P2", "P3"],
                out_size=self.config.desc_input_size,
                reduce_ratio=float(self.config.desc_input_size) /
                float(get_patch_size(self.config)),
            )

            # ----------------------------------------
            # Feature Descriptor
            #
            # The descriptor simply computes the descriptors, given the patch.
            self._build_module(
                module="desc",
                inputs=self.outputs["rot"],
                bypass=self.outputs["rot"],
                names=["P1", "P2", "P3"],
                skip=False,
            )

    def _build_module(self, module, inputs, bypass, names, skip, reuse=None,
                      test_only=False):
        """Subroutine for building each module"""

        is_first = True
        # _module = "kp"
        for name in names:
            # reuse if not the first time
            if not is_first:
                cur_reuse = True
            else:
                cur_reuse = reuse
            with tf.variable_scope(module, reuse=cur_reuse) as sc:
                if self.config.mean_std_type == 'batch':
                    sample_mean, sample_std = tf.nn.moments(
                        inputs[name], axes=(1, 2, 3), keep_dims=True)
                    cur_inputs = (inputs[name] - sample_mean) / sample_std
                elif self.config.mean_std_type == 'sample':
                    sample_mean, sample_std = tf.nn.moments(
                        inputs[name], axes=(0, 1, 2, 3), keep_dims=True)
                    cur_inputs = (inputs[name] - sample_mean) / sample_std
                else:
                    cur_inputs = (
                        (inputs[name] - self.mean[module]) /
                        self.std[module]
                    )
                if test_only:
                    is_training = False
                else:
                    is_training = self.is_training
                self.outputs[module][name] = self.modules[module].process(
                    inputs=cur_inputs,
                    bypass=bypass[name],
                    name=name,
                    skip=skip,
                    config=self.config,
                    is_training=is_training,
                )
                # Store variables if it is the first time
                if is_first:
                    self.params[module] = tf.get_collection(
                        tf.GraphKeys.TRAINABLE_VARIABLES, scope=sc.name)
                    self.allparams[module] = tf.get_collection(
                        tf.GraphKeys.GLOBAL_VARIABLES, scope=sc.name)
                    # Also append to the global list
                    self.params["joint"] += self.params[module]
                    self.allparams["joint"] += self.allparams[module]
                    # Mark that it is initialized
                    is_first = False

    def _build_st(self, module, xyz, cs, names, out_size,
                  reduce_ratio=None, do_rotate=False, do_reverse=False):
        """Subroutine for building spatial transformer"""

        for name in names:
            with tf.variable_scope(module):
                cur_inputs = self.inputs["patch"][name]
                # Get xy and cs
                if xyz is not None:
                    _xyz = xyz[name]["xyz"]
                else:
                    batch_size = tf.shape(cur_inputs)[0]
                    _xyz = tf.zeros((batch_size, 3))
                if cs is not None:
                    _cs = cs[name]["cs"]
                else:
                    _cs = None
                # transform coordinates
                # if do_rotate:
                #     _xyz[:2] = self.transform_xyz(_xyz,
                #                                   _cs,
                #                                   config.batch_size,
                #                                   reverse=do_reverse)
                input_size = get_tensor_shape(cur_inputs)[2]
                if reduce_ratio is None:
                    reduce_ratio = float(out_size) / float(input_size)
                # apply the spatial transformer
                self.outputs[module][name] = transformer(
                    U=cur_inputs,
                    # Get the output from the keypoint layer
                    theta=make_theta(xyz=_xyz, cs=_cs, rr=reduce_ratio),
                    out_size=(out_size, out_size),
                )

    def chain_cs(self, cs1, cs2, names):
        cs = {}
        for name in names:
            c1 = cs1[name]["cs"][:, 0]
            s1 = cs1[name]["cs"][:, 1]
            c2 = cs2[name]["cs"][:, 0]
            s2 = cs2[name]["cs"][:, 1]
            z = tf.zeros_like(c1)
            o = tf.ones_like(c1)
            mat1 = tf.transpose(tf.stack([[c1, -s1, z],
                                          [s1, c1, z],
                                          [z, z, o]]),
                                (2, 0, 1))
            mat2 = tf.transpose(tf.stack([[c2, -s2, z],
                                          [s2, c2, z],
                                          [z, z, o]]),
                                (2, 0, 1))
            joint = tf.matmul(mat1, mat2)
            cs[name] = {"cs": tf.stack(
                [joint[:, 0, 0], joint[:, 1, 0]],
                axis=1)}
        return cs

    def transform_xyz(self, xyz, cs, batch_size, scale, transpose, names):
        """Rotate a set of coordinates and apply scaling if necessary."""

        xyz_rot = {}
        for name in names:
            if cs is None:
                xyz_rot[name] = xyz[name] * scale
            else:
                c = cs[name]["cs"][:, 0]
                s = cs[name]["cs"][:, 1]
                z = tf.zeros_like(c)
                o = tf.ones_like(c)
                mat = tf.transpose(tf.stack([[c, -s, z],
                                             [s, c, z],
                                             [z, z, o]]),
                                   (2, 0, 1))
                xyz_rot[name] = tf.squeeze(
                    tf.matmul(mat,
                              tf.expand_dims(xyz[name], 2),
                              transpose_a=transpose),
                    axis=2) * scale
        return xyz_rot

    def transform_kp(self, kp, cs, batch_size, scale, transpose, names):
        """Rotate/scale keypoint coordinates while preserving the score."""

        # A bit inelegant but I'm tired
        kp_rot = {}
        for name in names:
            if cs is None:
                kp_rot[name] = {"xyz": kp[name]["xyz"] * scale,
                                "score": kp[name]["score"]}
            else:
                c = cs[name]["cs"][:, 0]
                s = cs[name]["cs"][:, 1]
                z = tf.zeros_like(c)
                o = tf.ones_like(c)
                mat = tf.transpose(tf.stack([[c, -s, z],
                                             [s, c, z],
                                             [z, z, o]]),
                                   (2, 0, 1))
                kp_rot[name] = {
                    "xyz": tf.squeeze(
                        tf.matmul(mat,
                                  tf.expand_dims(kp[name]["xyz"], 2),
                                  transpose_a=transpose),
                        axis=2) * scale,
                    "score": kp[name]["score"]}
        return kp_rot

    def _build_loss(self):
        """Build losses related to each subtask."""

        self.loss = {}

        # Indivisual loss components
        with tf.variable_scope("kp-overlap"):
            aug_rot = self.inputs["aug_rot"] if self.config.augment_rotations \
                else None
            gt = self.transform_xyz(self.inputs["xyz"],
                                    aug_rot,
                                    self.config.batch_size,
                                    self.scale_aug,
                                    transpose=True,
                                    names=["P1", "P2"])
            gt1 = gt["P1"]
            gt2 = gt["P2"]
            # gt1 = self.inputs["xyz"]["P1"] * self.scale_aug
            # gt2 = self.inputs["xyz"]["P2"] * self.scale_aug
            _loss_overlap = loss_overlap(
                kp_pos1=self.outputs["kp"]["P1"]["xyz"],
                gt_pos1=gt1,
                kp_pos2=self.outputs["kp"]["P2"]["xyz"],
                gt_pos2=gt2,
                r_base=self.r_base,
            )

        with tf.variable_scope("kp-classification"):
            _loss_classification = loss_classification(
                s1=self.outputs["kp"]["P1"]["score"],
                s2=self.outputs["kp"]["P2"]["score"],
                s3=self.outputs["kp"]["P3"]["score"],
                s4=self.outputs["kp"]["P4"]["score"],
            )

        with tf.variable_scope("desc-pair"):
            _loss_desc_pair = loss_desc_pair(
                d1=self.outputs["desc"]["P1"]["desc"],
                d2=self.outputs["desc"]["P2"]["desc"],
            )

        with tf.variable_scope("desc-non-pair"):
            if self.config.use_hardest_anchor:
                _loss_desc_non_pair = loss_desc_non_pair(
                    d1=self.outputs["desc"]["P1"]["desc"],
                    d2=self.outputs["desc"]["P2"]["desc"],
                    d3=self.outputs["desc"]["P3"]["desc"],
                    margin=self.config.alpha_margin,
                )
            else:
                _loss_desc_non_pair = loss_desc_non_pair(
                    d1=self.outputs["desc"]["P1"]["desc"],
                    d3=self.outputs["desc"]["P3"]["desc"],
                    margin=self.config.alpha_margin,
                )

        with tf.variable_scope("desc-triplet"):
            _loss_desc_triplet = loss_desc_triplet(
                d1=self.outputs["desc"]["P1"]["desc"],
                d2=self.outputs["desc"]["P2"]["desc"],
                d3=self.outputs["desc"]["P3"]["desc"],
                margin=self.config.triplet_loss_margin,
                mine_negative=self.config.use_hardest_anchor,
            )

        # Loss for each task
        self.loss["kp"] = (
            self.config.alpha_overlap * _loss_overlap +
            self.config.alpha_classification * _loss_classification
        )
        self.loss["ori"] = _loss_desc_pair
        if self.config.use_triplet_loss:
            self.loss["desc"] = _loss_desc_triplet
            self.loss["joint"] = (
                self.config.alpha_kp * _loss_classification +
                self.config.alpha_desc * _loss_desc_triplet
            )
        else:
            self.loss["desc"] = _loss_desc_pair + _loss_desc_non_pair
            self.loss["joint"] = (
                self.config.alpha_kp * _loss_classification +
                self.config.alpha_desc * _loss_desc_pair +
                self.config.alpha_desc * _loss_desc_non_pair
            )

        # Loss for weight decay
        #
        # LATER: we actually did not use it

        # Add summary for the subtask loss
        tf.summary.scalar(
            "losses/loss-" + self.config.subtask,
            tf.reduce_mean(self.loss[self.config.subtask]))

    def _build_optim(self):
        """Build the optimization op

        Note that we only construct it for the task at hand. This is to avoid
        having to deal with bypass layers

        """
        self.optim = {}

        # Optimizer depending on the option
        optimizer = self.config.optimizer
        learning_rate = self.config.learning_rate
        if optimizer == "sgd":
            optim = tf.train.GradientDescentOptimizer(learning_rate)
        elif optimizer == "adam":
            optim = tf.train.AdamOptimizer(learning_rate)
        elif optimizer == "rmsprop":
            optim = tf.train.RMSPropOptimizer(learning_rate)
        else:
            raise Exception("[!] Unknown optimizer: {}".format(optimizer))

        # All gradient computation should be done *after* the batchnorm update
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            # Compute gradient using the optimizer
            subtask = self.config.subtask
            loss = self.loss[subtask]
            params = self.params[subtask]
            # The subtask loss to optimize
            if subtask != "joint":
                # In case of other losses, simply do so
                grads_and_vars = optim.compute_gradients(loss, var_list=params)
            elif subtask == "joint":
                # In case of the joint optimization, be sure to treat
                # orientation in a different way.

                # for keypoint
                gv_kp = optim.compute_gradients(
                    self.loss["joint"], var_list=self.params["kp"])
                # for orientation
                gv_ori = optim.compute_gradients(
                    self.loss["ori"], var_list=self.params["ori"])
                # for descriptor
                gv_desc = optim.compute_gradients(
                    self.loss["joint"], var_list=self.params["desc"])

                # Concatenate the list
                # grads_and_vars = gv_kp + gv_ori + gv_desc
                grads_and_vars = []
                if 'kp' in self.config.finetune.split('+'):
                    grads_and_vars += gv_kp
                if 'ori' in self.config.finetune.split('+'):
                    grads_and_vars += gv_ori
                if 'desc' in self.config.finetune.split('+'):
                    grads_and_vars += gv_desc
                if len(grads_and_vars) == 0:
                    raise RuntimeError("Nothing to finetune? Check --finetune")

            else:
                raise ValueError("Wrong subtask {}".format(subtask))

            # Clip gradients if necessary
            if self.config.max_grad_norm > 0.0:
                new_grads_and_vars = []
                # check whether gradients contain large value (then clip), NaN
                # and InF
                for idx, (grad, var) in enumerate(grads_and_vars):
                    if grad is not None and var in params:
                        grad = tf.clip_by_norm(grad, self.config.max_grad_norm)
                        new_grads_and_vars.append((grad, var))
                grads_and_vars = new_grads_and_vars

            # Check numerics and report if something is going on. This will
            # make the backward pass stop and skip the batch
            if self.config.check_numerics is True:
                new_grads_and_vars = []
                # check whether gradients contain large value (then clip), NaN
                # and InF
                for idx, (grad, var) in enumerate(grads_and_vars):
                    if grad is not None and var in params:
                        grad = tf.check_numerics(
                            grad, "Numerical error in gradient for {}".format(
                                var.name))
                        new_grads_and_vars.append((grad, var))
                grads_and_vars = new_grads_and_vars

            # Summarize all gradients
            for grad, var in grads_and_vars:
                tf.summary.histogram(var.name + '/gradient', grad)

            # Make the optim op
            if len(grads_and_vars) > 0:
                self.optim[subtask] = optim.apply_gradients(grads_and_vars)

    def _get_feed_dict(self, subtask, cur_data):
        """Returns feed_dict"""

        #
        # To simplify things, we feed everything for now.
        #
        # LATER: make feed_dict less redundant.

        feed_dict = {}

        types = ["patch", "xyz", "angle"]
        if self.config.augment_rotations \
                or self.config.use_augmented_set:
            types += ["aug_rot"]

        for _type in types:
            for _name in ["P1", "P2", "P3", "P4"]:
                if _type == 'aug_rot':
                    key_cs = self.inputs[_type][_name]["cs"]
                    feed_dict[key_cs] = cur_data[_type][_name]["cs"]
                    key_angle = self.inputs[_type][_name]["angle"]
                    feed_dict[key_angle] = cur_data[_type][
                        _name]["angle"][..., None]
                else:
                    key = self.inputs[_type][_name]
                    feed_dict[key] = cur_data[_type][_name]

        return feed_dict

#
# lift.py ends here
