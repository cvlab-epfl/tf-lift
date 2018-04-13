# tester.py ---
#
# Filename: tester.py
# Description:
# Author: Kwang Moo Yi
# Maintainer:
# Created: Thu Jul  6 13:34:04 2017 (+0200)
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


import time
import os

import cv2
import h5py
import numpy as np
import tensorflow as tf

from datasets.test import Dataset
from networks.lift import Network
from six.moves import xrange
from utils import (IDX_ANGLE, XYZS2kpList, draw_XYZS_to_img, get_patch_size,
                   get_ratio_scale, get_XYZS_from_res_list, restore_network,
                   saveh5, saveKpListToTxt, update_affine, loadh5)


class Tester(object):
    """The Tester Class

    LATER: Clean up unecessary dictionaries
    LATER: Make a superclass for Tester and Trainer

    """

    def __init__(self, config, rng):
        self.config = config
        self.rng = rng

        # Open a tensorflow session. I like keeping things simple, so I don't
        # use a supervisor. I'm just going to do everything manually. I also
        # will just allow the gpu memory to grow
        tfconfig = tf.ConfigProto()
        tfconfig.gpu_options.allow_growth = True
        self.sess = tf.Session(config=tfconfig)

        # Create the dataset instance
        self.dataset = Dataset(self.config, rng)
        # Retrieve mean/std (yes it is hacky)
        logdir = os.path.join(self.config.logdir, self.config.subtask)
        if os.path.exists(os.path.join(logdir, "mean.h5")):
            training_mean = loadh5(os.path.join(logdir, "mean.h5"))
            training_std = loadh5(os.path.join(logdir, "std.h5"))
            print("[{}] Loaded input normalizers for testing".format(
                self.config.subtask))

            # Create the model instance
            self.network = Network(self.sess, self.config, self.dataset, {
                                   'mean': training_mean, 'std': training_std})
        else:
            self.network = Network(self.sess, self.config, self.dataset)
        # Make individual saver instances for each module.
        self.saver = {}
        self.best_val_loss = {}
        self.best_step = {}
        # Create the saver instance for both joint and the current subtask
        for _key in ["joint", self.config.subtask]:
            self.saver[_key] = tf.train.Saver(self.network.allparams[_key])

        # We have everything ready. We finalize and initialie the network here.
        self.sess.run(tf.global_variables_initializer())

    def run(self):

        subtask = self.config.subtask

        # Load the network weights for the module of interest
        print("-------------------------------------------------")
        print(" Loading Trained Network ")
        print("-------------------------------------------------")
        # Try loading the joint version, and then fall back to the current task
        # silently if failed.
        try:
            restore_res = restore_network(self, "joint")
        except:
            pass
        if not restore_res:
            restore_res = restore_network(self, subtask)
        if not restore_res:
            raise RuntimeError("Could not load network weights!")

        # Run the appropriate compute function
        print("-------------------------------------------------")
        print(" Testing ")
        print("-------------------------------------------------")

        eval("self._compute_{}()".format(subtask))

    def _compute_kp(self):
        """Compute Keypoints.

        LATER: Clean up code

        """

        total_time = 0.0

        # Read image
        image_color, image_gray, load_prep_time = self.dataset.load_image()

        # check size
        image_height = image_gray.shape[0]
        image_width = image_gray.shape[1]

        # Multiscale Testing
        scl_intv = self.config.test_scl_intv
        # min_scale_log2 = 1  # min scale = 2
        # max_scale_log2 = 4  # max scale = 16
        min_scale_log2 = self.config.test_min_scale_log2
        max_scale_log2 = self.config.test_max_scale_log2
        # Test starting with double scale if small image
        min_hw = np.min(image_gray.shape[:2])
        # for the case of testing on same scale, do not double scale
        if min_hw <= 1600 and min_scale_log2!=max_scale_log2:
            print("INFO: Testing double scale")
            min_scale_log2 -= 1
        # range of scales to check
        num_division = (max_scale_log2 - min_scale_log2) * (scl_intv + 1) + 1
        scales_to_test = 2**np.linspace(min_scale_log2, max_scale_log2,
                                        num_division)

        # convert scale to image resizes
        resize_to_test = ((float(self.config.kp_input_size - 1) / 2.0) /
                          (get_ratio_scale(self.config) * scales_to_test))

        # check if resize is valid
        min_hw_after_resize = resize_to_test * np.min(image_gray.shape[:2])
        is_resize_valid = min_hw_after_resize > self.config.kp_filter_size + 1

        # if there are invalid scales and resizes
        if not np.prod(is_resize_valid):
            # find first invalid
            # first_invalid = np.where(True - is_resize_valid)[0][0]
            first_invalid = np.where(~is_resize_valid)[0][0]

            # remove scales from testing
            scales_to_test = scales_to_test[:first_invalid]
            resize_to_test = resize_to_test[:first_invalid]

        print('resize to test is {}'.format(resize_to_test))
        print('scales to test is {}'.format(scales_to_test))

        # Run for each scale
        test_res_list = []
        for resize in resize_to_test:

            # resize according to how we extracted patches when training
            new_height = np.cast['int'](np.round(image_height * resize))
            new_width = np.cast['int'](np.round(image_width * resize))
            start_time = time.clock()
            image = cv2.resize(image_gray, (new_width, new_height))
            end_time = time.clock()
            resize_time = (end_time - start_time) * 1000.0
            print("Time taken to resize image is {}ms".format(
                resize_time
            ))
            total_time += resize_time

            # run test
            # LATER: Compatibility with the previous implementations
            start_time = time.clock()

            # Run the network to get the scoremap (the valid region only)
            scoremap = None
            if self.config.test_kp_use_tensorflow:
                scoremap = self.network.test(
                    self.config.subtask,
                    image.reshape(1, new_height, new_width, 1)
                ).squeeze()
            else:
                # OpenCV Version
                raise NotImplementedError(
                    "TODO: Implement OpenCV Version")

            end_time = time.clock()
            compute_time = (end_time - start_time) * 1000.0
            print("Time taken for image size {}"
                  " is {} milliseconds".format(
                      image.shape, compute_time))

            total_time += compute_time

            # pad invalid regions and add to list
            start_time = time.clock()
            test_res_list.append(
                np.pad(scoremap, int((self.config.kp_filter_size - 1) / 2),
                       mode='constant',
                       constant_values=-np.inf)
            )
            end_time = time.clock()
            pad_time = (end_time - start_time) * 1000.0
            print("Time taken for padding and stacking is {} ms".format(
                pad_time
            ))
            total_time += pad_time

        # ------------------------------------------------------------------------
        # Non-max suppresion and draw.

        # The nonmax suppression implemented here is very very slow. Consider
        # this as just a proof of concept implementation as of now.

        # Standard nearby : nonmax will check approximately the same area as
        # descriptor support region.
        nearby = int(np.round(
            (0.5 * (self.config.kp_input_size - 1.0) *
             float(self.config.desc_input_size) /
             float(get_patch_size(self.config)))
        ))
        fNearbyRatio = self.config.test_nearby_ratio
        # Multiply by quarter to compensate
        fNearbyRatio *= 0.25
        nearby = int(np.round(nearby * fNearbyRatio))
        nearby = max(nearby, 1)

        nms_intv = self.config.test_nms_intv
        edge_th = self.config.test_edge_th

        print("Performing NMS")
        start_time = time.clock()
        res_list = test_res_list
        # check whether the return result for socre is right
#        print(res_list[0][400:500,300:400])
        XYZS = get_XYZS_from_res_list(
            res_list, resize_to_test, scales_to_test, nearby, edge_th,
            scl_intv, nms_intv, do_interpolation=True,
        )
        end_time = time.clock()
        XYZS = XYZS[:self.config.test_num_keypoint]

        # For debugging
        # TODO: Remove below
        draw_XYZS_to_img(XYZS, image_color, self.config.test_out_file + '.jpg')

        nms_time = (end_time - start_time) * 1000.0
        print("NMS time is {} ms".format(nms_time))
        total_time += nms_time
        print("Total time for detection is {} ms".format(total_time))
        # if bPrintTime:
        #     # Also print to a file by appending
        #     with open("../timing-code/timing.txt", "a") as timing_file:
        #         print("------ Keypoint Timing ------\n"
        #               "NMS time is {} ms\n"
        #               "Total time is {} ms\n".format(
        #                   nms_time, total_time
        #               ),
        #               file=timing_file)

        # # resize score to original image size
        # res_list = [cv2.resize(score,
        #                        (image_width, image_height),
        #                        interpolation=cv2.INTER_NEAREST)
        #             for score in test_res_list]
        # # make as np array
        # res_scores = np.asarray(res_list)
        # with h5py.File('test/scores.h5', 'w') as score_file:
        #     score_file['score'] = res_scores

        # ------------------------------------------------------------------------
        # Save as keypoint file to be used by the oxford thing
        print("Turning into kp_list")
        kp_list = XYZS2kpList(XYZS)  # note that this is already sorted

        # ------------------------------------------------------------------------
        # LATER: take care of the orientations somehow...
        # # Also compute angles with the SIFT method, since the keypoint
        # # component alone has no orientations.
        # print("Recomputing Orientations")
        # new_kp_list, _ = recomputeOrientation(image_gray, kp_list,
        #                                       bSingleOrientation=True)

        print("Saving to txt")
        saveKpListToTxt(kp_list, None, self.config.test_out_file)

    def _compute_ori(self):
        """Compute Orientations """

        total_time = 0.0

        # Read image
        start_time = time.clock()
        cur_data = self.dataset.load_data()
        end_time = time.clock()
        load_time = (end_time - start_time) * 1000.0
        print("Time taken to load patches is {} ms".format(
            load_time
        ))
        total_time += load_time

        # -------------------------------------------------------------------------
        # Test using the test function
        start_time = time.clock()
        oris = self._test_multibatch(cur_data)
        end_time = time.clock()
        compute_time = (end_time - start_time) * 1000.0
        print("Time taken to compute is {} ms".format(
            compute_time
        ))
        total_time += compute_time

        # update keypoints and save as new
        start_time = time.clock()
        kps = cur_data["kps"]
        for idxkp in xrange(len(kps)):
            kps[idxkp][IDX_ANGLE] = oris[idxkp] * 180.0 / np.pi % 360.0
            kps[idxkp] = update_affine(kps[idxkp])
        end_time = time.clock()
        update_time = (end_time - start_time) * 1000.0
        print("Time taken to update is {} ms".format(
            update_time
        ))
        total_time += update_time
        print("Total time for orientation is {} ms".format(total_time))

        # save as new keypoints
        saveKpListToTxt(
            kps, self.config.test_kp_file, self.config.test_out_file)

    def _compute_desc(self):
        """Compute Descriptors """

        total_time = 0.0

        # Read image
        start_time = time.clock()
        cur_data = self.dataset.load_data()
        end_time = time.clock()
        load_time = (end_time - start_time) * 1000.0
        print("Time taken to load patches is {} ms".format(
            load_time
        ))
        total_time += load_time

        # import IPython
        # IPython.embed()

        # -------------------------------------------------------------------------
        # Test using the test function
        start_time = time.clock()
        descs = self._test_multibatch(cur_data)
        end_time = time.clock()
        compute_time = (end_time - start_time) * 1000.0
        print("Time taken to compute is {} ms".format(
            compute_time
        ))
        total_time += compute_time
        print("Total time for descriptor is {} ms".format(total_time))

        # Overwrite angle
        kps = cur_data["kps"].copy()
        kps[:, 3] = cur_data["angle"][:, 0]

        # Save as h5 file
        save_dict = {}
        # save_dict['keypoints'] = cur_data["kps"]
        save_dict['keypoints'] = kps
        save_dict['descriptors'] = descs

        saveh5(save_dict, self.config.test_out_file)

    def _test_multibatch(self, cur_data):
        """A sub test routine.

        We do this since the spatial transformer implementation in tensorflow
        does not like undetermined batch sizes.

        LATER: Bypass the spatial transformer...somehow
        LATER: Fix the multibatch testing

        """

        subtask = self.config.subtask
        batch_size = self.config.batch_size
        num_patch = len(cur_data["patch"])
        num_batch = int(np.ceil(float(num_patch) / float(batch_size)))
        # Initialize the batch items
        cur_batch = {}
        for _key in cur_data:
            cur_batch[_key] = np.zeros_like(cur_data[_key][:batch_size])

        # Do muiltiple times
        res = []
        for _idx_batch in xrange(num_batch):
            # start of the batch
            bs = _idx_batch * batch_size
            # end of the batch
            be = min(num_patch, (_idx_batch + 1) * batch_size)
            # number of elements in batch
            bn = be - bs
            for _key in cur_data:
                cur_batch[_key][:bn] = cur_data[_key][bs:be]
            cur_res = self.network.test(subtask, cur_batch).squeeze()[:bn]
            # Append
            res.append(cur_res)

        return np.concatenate(res, axis=0)

#
# tester.py ends here
