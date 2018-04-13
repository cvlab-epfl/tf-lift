# test.py ---
#
# Filename: test.py
# Description:
# Author: Kwang Moo Yi
# Maintainer:
# Created: Thu Jul  6 13:43:44 2017 (+0200)
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

import os
import time
from copy import deepcopy

import cv2
import numpy as np

from datasets.eccv2016.helper import load_patches
from utils import IDX_ANGLE, get_patch_size, get_ratio_scale, loadKpListFromTxt


class Dataset(object):
    """The Dataset Module


    """

    def __init__(self, config, rng):
        self.config = config
        self.rng = rng

    def load_image(self):
        """ Returns the image to work with """

        image_file_name = self.config.test_img_file

        # ------------------------------------------------------------------------
        # Run learned network
        start_time = time.clock()
        # resize_scale = 1.0
        # If there is not gray image, load the color one and convert to gray
        # read the image
        if os.path.exists(image_file_name.replace(
                "image_color", "image_gray"
        )) and "image_color" in image_file_name:
            image_gray = cv2.imread(image_file_name.replace(
                "image_color", "image_gray"
            ), 0)
            image_color = deepcopy(image_gray)
            image_resized = image_gray
            # ratio_h = float(
            #     image_resized.shape[0]) / float(image_gray.shape[0])
            # ratio_w = float(
            #     image_resized.shape[1]) / float(image_gray.shape[1])
        else:
            # read the image
            image_color = cv2.imread(image_file_name)
            image_resized = image_color
            # ratio_h = float(
            #     image_resized.shape[0]) / float(image_color.shape[0])
            # ratio_w = float(
            #     image_resized.shape[1]) / float(image_color.shape[1])
            image_gray = cv2.cvtColor(
                image_resized,
                cv2.COLOR_BGR2GRAY).astype("float32")

        assert len(image_gray.shape) == 2

        end_time = time.clock()
        load_prep_time = (end_time - start_time) * 1000.0
        print("Time taken to read and prepare the image is {} ms".format(
            load_prep_time
        ))

        return image_color, image_gray, load_prep_time

    def load_data(self):
        """Returns the patch, given the keypoint structure

        LATER: Cleanup. We currently re-use the utils we had from data
               extraction.

        """

        # Load image
        img = cv2.imread(self.config.test_img_file)
        # If color image, turn it to gray
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        in_dim = 1

        # Load keypoints
        kp = np.asarray(loadKpListFromTxt(self.config.test_kp_file))

        # Use load patches function
        # Assign dummy values to y, ID, angle
        y = np.zeros((len(kp),))
        ID = np.zeros((len(kp),), dtype='int64')
        # angle = np.zeros((len(kp),))
        angle = np.pi / 180.0 * kp[:, IDX_ANGLE]  # store angle in radians

        # load patches with id (drop out of boundary)
        bPerturb = False
        fPerturbInfo = np.zeros((3,))
        dataset = load_patches(img, kp, y, ID, angle,
                               get_ratio_scale(self.config), 1.0,
                               int(get_patch_size(self.config)),
                               int(self.config.desc_input_size), in_dim,
                               bPerturb, fPerturbInfo, bReturnCoords=True,
                               is_test=True)

        # Change old dataset return structure to necessary data
        x = dataset[0]
        # y = dataset[1]
        # ID = dataset[2]
        pos = dataset[3]
        angle = dataset[4]
        coords = dataset[5]

        # Return the dictionary structure
        cur_data = {}
        cur_data["patch"] = np.transpose(x, (0, 2, 3, 1))  # In NHWC
        cur_data["kps"] = coords
        cur_data["xyz"] = pos
        # Make sure that angle is a Nx1 vector
        cur_data["angle"] = np.reshape(angle, (-1, 1))

        return cur_data

#
# test.py ends here
