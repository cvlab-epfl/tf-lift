# eccv.py ---
#
# Filename: eccv.py
# Description:
# Author: Kwang Moo Yi
# Maintainer: Kwang Moo Yi
# Created: Fri Feb 26 12:39:54 2016 (+0100)
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
# Thu 29 Jun 14:52:41 CEST 2017, Kwang Moo Yi
#
# - Adaptation to be used inside the modules
# 
# Mon 26 Jun 13:52:37 CEST 2017, Kwang Moo Yi
#
# - Runs differently based on user name. Will use cvlab environment if possible
# - NFS safe lock is now re-enabled to prevent race condition.
#

# Code:


# -----------------------------------------------------------------------------
# Imports
from __future__ import print_function

import hashlib
import multiprocessing as mp
import os
import shutil
import sys
import time
from datetime import timedelta
from inspect import currentframe, getframeinfo

import cv2
import h5py
import numpy as np
import six
from flufl.lock import Lock

# from Utils.dataset_tools.helper import (load_patches,
#                                        random_mine_non_kp_with_2d_distance,
#                                        random_mine_non_kp_with_3d_blocking)
from datasets.eccv2016.helper import (load_patches,
                                      random_mine_non_kp_with_2d_distance,
                                      random_mine_non_kp_with_3d_blocking)
from six.moves import xrange
# from Utils.custom_types import pathConfig
from datasets.eccv2016.custom_types import pathConfig
from utils import loadh5, saveh5

# ratio of cpu cores this script is allowed to use
ratio_CPU = 0.5


# -----------------------------------------------------------------------------
# Dataset class
def createDump(args):

    idx_jpg, jpg_file, train_data_dir, dump_data_dir, tmp_patch_dir, \
        scale_hist, scale_hist_c, out_dim, param, queue = args

    # queue for monitoring
    if queue is not None:
        queue.put(idx_jpg)

    final_dump_file_name = tmp_patch_dir + jpg_file.replace(".jpg", ".h5")
    if not os.path.exists(final_dump_file_name):
        # load image
        bUseColorImage = getattr(param.patch, "bUseColorImage", False)
        if not bUseColorImage:
            bUseDebugImage = getattr(param.patch, "bUseDebugImage", False)
            if not bUseDebugImage:
                img = cv2.cvtColor(cv2.imread(train_data_dir + jpg_file),
                                   cv2.COLOR_BGR2GRAY)
            else:
                # debug image contains keypoints survive the SfM
                # pipeline (blue) and the rest (in red)
                img = cv2.cvtColor(cv2.imread(train_data_dir + jpg_file),
                                   cv2.COLOR_BGR2GRAY)
                debug_jpg = train_data_dir + jpg_file.replace(
                    ".jpg", "-kp-minsc-" + str(param.dataset.fMinKpSize) +
                    ".jpg")
                imgd = cv2.cvtColor(cv2.imread(debug_jpg), cv2.COLOR_BGR2GRAY)
                img = cv2.resize(imgd, img.shape[:2][::-1])

            in_dim = 1
        else:
            img = cv2.imread(train_data_dir + jpg_file)
            in_dim = 3
            assert(img.shape[-1] == in_dim)

        # ----------------------------------------
        # load kp data
        # Note that data is in order of [x,y,scale,angle,pointid,setid]]

        # For positive: select randomly from "valid_keypoints"
        pos_kp_file_name = train_data_dir + jpg_file.replace(
            ".jpg", "-kp-minsc-" + str(param.dataset.fMinKpSize) + ".h5")
        with h5py.File(pos_kp_file_name, "r") as h5file:
            sfm_kp = np.asarray(h5file["valid_keypoints"], dtype="float")
            non_sfm_kp = np.asarray(h5file["other_keypoints"], dtype="float")
            # add two dummy fields to non_sfm since they don"t have id
            # and group. THis means pointid,setid are set to 1 for non_sfm_kp
            non_sfm_kp = np.concatenate(
                [non_sfm_kp, -np.ones((non_sfm_kp.shape[0], 2))],
                axis=1
            )

        pos_kp = np.concatenate((sfm_kp, np.ones((sfm_kp.shape[0], 1),
                                                 dtype="float")), axis=1)
        # Select subset for positives (assuming we have same coordinates for
        # all image)
        dump_file_name = dump_data_dir + jpg_file.replace(".jpg",
                                                          "_idxPosSel.h5")

        # if dump file not exist, create it; otherwise load it
        if not os.path.exists(dump_file_name):
            # idxPosShuffle = np.argsort(pos_kp[3,:])[::-1] # sort backwards
            idxPosShuffle = np.random.permutation(
                len(pos_kp))  # random shuffle
            pos_2_keep = len(idxPosShuffle)
            if param.dataset.nPosPerImg > 0:
                pos_2_keep = min(pos_2_keep, param.dataset.nPosPerImg)
            idxPosSel = idxPosShuffle[:pos_2_keep]  # shuffle the points
            to_save = {"saveval": idxPosSel}
            saveh5(to_save, dump_file_name)
        else:
            to_load = loadh5(dump_file_name)
            idxPosSel = to_load["saveval"]
        pos_kp = pos_kp[idxPosSel]

        # negative sampling:
        #
        # 1) only use Sfm points, too close keypoints will be rejected as
        # negative pair (because of high potential overlapping)
        #
        # 2) Use all SIFT points, when the overlapping of two feature context
        # windows is larger than a threshold, it will be rejected as a negative
        # pair (because it shares too much common regions..). In this step, we
        # concatenate sfm_kp and non_sfm_kp to form keypoint class.
        neg_mine_method = getattr(param.patch, "sNegMineMethod",
                                  "use_only_SfM_points")
        if neg_mine_method == "use_only_SfM_points":
            # where is
            neg_kp = random_mine_non_kp_with_2d_distance(
                img, sfm_kp, scale_hist, scale_hist_c, param)
        elif neg_mine_method == "use_all_SIFT_points":
            max_iter = getattr(param.patch, "nMaxRandomNegMineIter", 100)
            sift_kp = np.concatenate([sfm_kp, non_sfm_kp], axis=0)
            neg_kp = random_mine_non_kp_with_3d_blocking(
                img, sift_kp, scale_hist, scale_hist_c, param,
                max_iter=max_iter)
        else:
            raise ValueError("Mining method {} is not supported!"
                             "".format(neg_mine_method))
        # add another dim indicating good or bad
        neg_kp = np.concatenate((neg_kp, np.zeros((len(neg_kp), 1),
                                                  dtype="float")), axis=1)
        # concatenate negative and positives
        kp = np.concatenate((pos_kp, neg_kp), axis=0)

        # Retrive target values, 1 for pos and 0 for neg
        y = kp[:, 6]

        # Retrieve angles
        angle = kp[:, 3]

        # Assign ID to keypoints (for sfm points, ID = 3d point ind; for non
        # sfm kp, ID = -1)
        ID = kp[:, 4]

        # load patches with id (drop out of boundary)
        bPerturb = getattr(param.patch, "bPerturb", False)
        fPerturbInfo = getattr(param.patch, "fPerturbInfo", np.zeros((3,)))
        nAugmentedRotations = getattr(param.patch, "nAugmentedRotations", 1)
        fAugmentRange = getattr(param.patch, "fAugmentRange", 0)
        fAugmentCenterRandStrength = getattr(
            param.patch, "fAugmentCenterRandStrength", 0
        )
        sAugmentCenterRandMethod = getattr(
            param.patch, "sAugmentCenterRandMethod", "uniform"
        )

        cur_data_set = load_patches(
            img, kp, y, ID, angle, param.patch.fRatioScale,
            param.patch.fMaxScale, param.patch.nPatchSize,
            param.model.nDescInputSize, in_dim, bPerturb, fPerturbInfo,
            bReturnCoords=True,
            nAugmentedRotations=nAugmentedRotations,
            fAugmentRange=fAugmentRange,
            fAugmentCenterRandStrength=fAugmentCenterRandStrength,
            sAugmentCenterRandMethod=sAugmentCenterRandMethod,
            nPatchSizeAug=param.patch.nPatchSizeAug,
        )
        # save dump as dictionary using saveh5
        # from here Kernel died, because of not finding "MKL_intel_thread.dll"

        # pdb.set_trace()
        # here the data are saved in tmp_dump file, keys are numbers [0,1..]
        tmpdict = dict(
            (str(_idx), np.asarray(_data)) for _idx, _data in
            zip(np.arange(len(cur_data_set)), cur_data_set)
        )
        saveh5(tmpdict, final_dump_file_name)

    return idx_jpg


class data_obj(object):
    """ Dataset Object class.

    Implementation of the dataset object
    """

    def __init__(self, param, mode="train"):

        # Set parameters
        self.out_dim = 1        # a single regressor output

        # Load data
        # self.x = None           # data (patches) to be used for learning [N,
        #                         # channel, w, h]
        # self.y = None           # label/target to be learned
        # self.ID = None          # id of the data for manifold regularization
        self.load_data(param, mode)

        # Set parameters
        self.num_channel = self.x.shape[1]  # single channel image
        # patch width == patch height (28x28)
        self.patch_height = self.x.shape[2]
        self.patch_width = self.x.shape[3]

        self

    def load_data(self, param, mode):

        print(" --------------------------------------------------- ")
        print(" ECCV Data Module ")
        print(" --------------------------------------------------- ")

        # for each dataset item in the list
        id_base = 0
        for idxSet in xrange(len(param.dataset.trainSetList)):

            pathconf = pathConfig()
            pathconf.setupTrain(param, param.dataset.trainSetList[idxSet])

            # Save the hash, for the pairs folder
            # TODO make this work with multiple datasets
            self.hash = '/'.join(pathconf.patch_dump.split('/')[-3:])

            # load the data from one dataset (multi dataset can be used)
            cur_data = self.load_data_for_set(pathconf, param, mode)
            fixed_id = cur_data[2] + (cur_data[2] >= 0) * id_base
            if idxSet == 0:
                self.x = cur_data[0]
                self.y = cur_data[1]
                self.ID = fixed_id
                self.pos = cur_data[3]
                self.angle = cur_data[4]
                self.coords = cur_data[5]
            else:
                self.x = np.concatenate([self.x, cur_data[0]])
                self.y = np.concatenate([self.y, cur_data[1]])
                self.ID = np.concatenate([self.ID, fixed_id])
                self.pos = np.concatenate([self.pos, cur_data[3]])
                self.angle = np.concatenate([self.angle, cur_data[4]])
                self.coords = np.concatenate([self.coords, cur_data[5]])

            id_base = np.max(self.ID) + 1

    def load_data_for_set(self, pathconf, param, mode):

        # ----------------------------------------------------------------------
        # Train, Validation, and Test
        # mlab = matlab.engine.start_matlab()

        # Read from pathconf
        # Original implementation
        train_data_dir = pathconf.dataset
        dump_data_dir = pathconf.train_dump
        dump_patch_dir = pathconf.patch_dump
        # local (or volatile) copy of the dump data
        tmp_patch_dir = pathconf.volatile_patch_dump

        print("train_data_dir = {}".format(train_data_dir))
        print("dump_data_dir = {}".format(dump_data_dir))
        print("dump_patch_dir = {}".format(dump_patch_dir))
        print("tmp_patch_dir = {}".format(tmp_patch_dir))

        if not os.path.exists(dump_data_dir):
            os.makedirs(dump_data_dir)
        if not os.path.exists(dump_patch_dir):
            os.makedirs(dump_patch_dir)
        if not os.path.exists(tmp_patch_dir):
            os.makedirs(tmp_patch_dir)

        # Check if we have the big h5 file ready
        big_file_name = dump_patch_dir + mode + "-data-chunked.h5"
        # if os.getenv("MLTEST_DEBUG", default=""):
        #     import pdb
        #     pdb.set_trace()

        # Mutex lock
        #
        # We will create an nfs-safe lock file in a temporary directory to
        # prevent our script from using corrupted, or data that is still being
        # generated. This allows us to launch multiple instances at the same
        # time, and allow only a single instance to generate the big_file.
        if not os.path.exists(".locks"):
            os.makedirs(".locks")
        check_lock_file = ".locks/" + \
            hashlib.md5(big_file_name.encode()).hexdigest()
        if os.name == "posix":
            check_lock = Lock(check_lock_file)
            check_lock.lifetime = timedelta(days=2)
            frameinfo = getframeinfo(currentframe())
            print("-- {}/{}: waiting to obtain lock --".format(
                frameinfo.filename, frameinfo.lineno))
            check_lock.lock()
            print(">> obtained lock for posix system<<")
        elif os.name == "nt":
            import filelock
            check_lock = filelock.FileLock(check_lock_file)
            check_lock.timeout = 2000
            check_lock.acquire()
            if check_lock.is_locked:
                print(">> obtained lock for windows system <<")
        else:
            print("Unknown operating system, lock unavailable")

        if not os.path.exists(big_file_name):

            if not os.path.exists(dump_patch_dir + mode + "-data.h5"):

                # Read scale histogram
                hist_file = h5py.File(train_data_dir +
                                      "scales-histogram-minsc-" +
                                      str(param.dataset.fMinKpSize) + ".h5",
                                      "r")
                scale_hist = np.asarray(hist_file["histogram_bins"]).flatten()
                scale_hist /= np.sum(scale_hist)
                scale_hist_c = np.asarray(
                    hist_file["histogram_centers"]
                ).flatten()

                # Read list of images from split files
                split_name = ""
                split_name += str(param.dataset.nTrainPercent) + "-"
                split_name += str(param.dataset.nValidPercent) + "-"
                split_name += str(param.dataset.nTestPercent) + "-"
                if mode == "train":
                    split_name += "train-"
                elif mode == "valid":
                    split_name += "val-"
                elif mode == "test":
                    split_name += "test-"
                split_file_name = train_data_dir + "split-" \
                    + split_name + "minsc-" \
                    + str(param.dataset.fMinKpSize) + ".h.txt"
                list_jpg_file = []
                for file_name in list(
                        np.loadtxt(split_file_name, dtype=bytes)
                ):
                    list_jpg_file += [
                        file_name.decode("utf-8").replace(
                            "-kp-minsc-" + str(param.dataset.fMinKpSize),
                            ".jpg")
                    ]

                # -------------------------------------------------
                # Create dumps in parallel
                # I am lazy so create arguments in loop lol
                pool_arg = [None] * len(list_jpg_file)
                for idx_jpg in six.moves.xrange(len(list_jpg_file)):
                    pool_arg[idx_jpg] = (idx_jpg,
                                         list_jpg_file[idx_jpg],
                                         train_data_dir,
                                         dump_data_dir, tmp_patch_dir,
                                         scale_hist, scale_hist_c,
                                         self.out_dim, param)

                # if true, use multi thread, otherwise use only single thread
                if True:
                    number_of_process = int(ratio_CPU * mp.cpu_count())
                    pool = mp.Pool(processes=number_of_process)
                    manager = mp.Manager()
                    queue = manager.Queue()
                    for idx_jpg in six.moves.xrange(len(list_jpg_file)):
                        pool_arg[idx_jpg] = pool_arg[idx_jpg] + (queue,)
                    # map async
                    pool_res = pool.map_async(createDump, pool_arg)
                    # monitor loop
                    while True:
                        if pool_res.ready():
                            # print("")
                            break
                        else:
                            size = queue.qsize()
                            print("\r -- " + mode + ": Processing image "
                                  "{}/{}".format(size, len(list_jpg_file)),
                                  end="")
                            sys.stdout.flush()
                            time.sleep(1)
                    pool.close()
                    pool.join()
                    print("\r -- " + mode + ": Finished Processing Images!")
                # for debugging, if multi thread is used, then it is difficult
                # to debug
                else:
                    for idx_jpg in six.moves.xrange(len(list_jpg_file)):
                        pool_arg[idx_jpg] = pool_arg[idx_jpg] + (None,)
                    for idx_jpg in six.moves.xrange(len(list_jpg_file)):
                        createDump(pool_arg[idx_jpg])
                        print("\r -- " + mode + ": Processing image "
                              "{}/{}".format(idx_jpg + 1, len(list_jpg_file)),
                              end="")
                        sys.stdout.flush()
                    print("\r -- " + mode + ": Finished Processing Images!")
                # -------------------------------------------------

                # # --------------------
                # # use single thread for simplify debugging
                # for idx_jpg in six.moves.xrange(len(list_jpg_file)):
                #     pool_arg[idx_jpg] = pool_arg[idx_jpg] + (None,)
                # for idx_jpg in six.moves.xrange(len(list_jpg_file)):
                #     createDump(pool_arg[idx_jpg])
                #     print("\r -- " + mode + ": Processing image "
                #           "{}/{}".format(idx_jpg + 1, len(list_jpg_file)),
                #           end="")
                #     sys.stdout.flush()
                # print("\r -- " + mode + ": Finished Processing Images!")

                # ------------------------------------------------------------------
                # Use only valid indices to ascertain mutual exclusiveness
                id_file_name = train_data_dir + "split-"
                id_file_name += str(param.dataset.nTrainPercent) + "-"
                id_file_name += str(param.dataset.nValidPercent) + "-"
                id_file_name += str(param.dataset.nTestPercent) + "-"
                id_file_name += ("minsc-" +
                                 str(param.dataset.fMinKpSize) +
                                 ".h5")

                if mode == "train":
                    id_key = "indices_train"
                elif mode == "valid":
                    id_key = "indices_val"
                elif mode == "test":
                    id_key = "indices_test"

                with h5py.File(id_file_name, "r") as id_file:
                    id_2_keep = np.asarray(id_file[id_key])

                # ind_2_keep = np.in1d(dataset[2], id_2_keep)
                # ind_2_keep += dataset[2] < 0

                # loop through files to figure out how many valid items we have
#                pdb.set_trace() # for tracking of the dataset

                num_valid = 0
                for idx_jpg in six.moves.xrange(len(list_jpg_file)):

                    jpg_file = list_jpg_file[idx_jpg]

                    print("\r -- " + mode + ": "
                          "Reading dumps to figure out number of valid "
                          "{}/{}".format(idx_jpg + 1, len(list_jpg_file)),
                          end="")
                    sys.stdout.flush()

                    # Load created dump
                    final_dump_file_name = tmp_patch_dir \
                        + jpg_file.replace(".jpg", ".h5")
                    # Use loadh5 and turn it back to original cur_data_set
                    with h5py.File(final_dump_file_name, "r") as dump_file:
                        cur_ids = dump_file["2"].value

                    # Find cur valid by looking at id_2_keep
                    cur_valid = np.in1d(cur_ids, id_2_keep)
                    # Add all negative labels as valid (neg data)
                    cur_valid += cur_ids < 0

                    # Sum it up
                    num_valid += np.sum(cur_valid)

                print("\n -- " + mode + ": "
                      "Found {} valid data points from {} files"
                      "".format(num_valid, len(list_jpg_file)))

                # Get the first data to simply check the shape
                tmp_dump_file_name = tmp_patch_dir \
                    + list_jpg_file[0].replace(".jpg", ".h5")
                with h5py.File(tmp_dump_file_name, "r") as dump_file:
                    dataset_shape = []
                    dataset_type = []
                    for _idx in six.moves.xrange(len(dump_file.keys())):
                        dataset_shape += [dump_file[str(_idx)].shape]
                        dataset_type += [dump_file[str(_idx)].dtype]

                # create and save the large dataset chunk
                with h5py.File(big_file_name, "w-") as big_file:
                    big_file["time_stamp"] = np.asarray(time.localtime())
                    name_list = ["x", "y", "ID", "pos", "angle", "coords"]
                    # create the dataset storage chunk
                    for __i in six.moves.xrange(len(dataset_shape)):
                        big_file.create_dataset(
                            name_list[__i],
                            (num_valid,) + dataset_shape[__i][1:],
                            chunks=(1,) + dataset_shape[__i][1:],
                            maxshape=(
                                (num_valid,) + dataset_shape[__i][1:]
                            ),
                            dtype=dataset_type[__i]
                        )
                    # loop through the file to save to a big chunk
                    save_base = 0
                    for idx_jpg in six.moves.xrange(len(list_jpg_file)):

                        jpg_file = list_jpg_file[idx_jpg]

                        print("\r -- " + mode + ": "
                              "Saving the data to the big dump "
                              "{}/{}".format(idx_jpg + 1, len(list_jpg_file)),
                              end="")
                        sys.stdout.flush()

                        # Load created dump
                        final_dump_file_name = tmp_patch_dir \
                            + jpg_file.replace(".jpg", ".h5")
                        # Use loadh5 and turn it back to original cur_data_set
                        tmpdict = loadh5(final_dump_file_name)
                        cur_data_set = tuple([
                            tmpdict[str(_idx)] for _idx in
                            range(len(tmpdict.keys()))
                        ])
                        # Find cur valid by looking at id_2_keep
                        cur_valid = np.in1d(cur_data_set[2], id_2_keep)
                        # Add all negative labels as valid (neg data)
                        cur_valid += cur_data_set[2] < 0
                        for __i in six.moves.xrange(len(dataset_shape)):
                            big_file[name_list[__i]][
                                save_base:save_base + np.sum(cur_valid)
                            ] = cur_data_set[__i][cur_valid]
                        # Move base to the next chunk
                        save_base += np.sum(cur_valid)

                    # Assert that we saved all
                    assert save_base == num_valid

                print("\n -- " + mode + ": "
                      "Done saving {} valid data points from {} files"
                      "".format(num_valid, len(list_jpg_file)))

                # --------------------------------------------------
                #  Cleanup dump
                for idx_jpg in six.moves.xrange(len(list_jpg_file)):

                    jpg_file = list_jpg_file[idx_jpg]

                    print("\r -- " + mode + ": "
                          "Removing dump "
                          "{}/{}".format(idx_jpg + 1, len(list_jpg_file)),
                          end="")
                    sys.stdout.flush()

                    # Delete dump
                    final_dump_file_name = tmp_patch_dir \
                        + jpg_file.replace(".jpg", ".h5")
                    os.remove(final_dump_file_name)

                print("\r -- " + mode + ": "
                      "Cleaned up dumps! "
                      "Local dump is now clean!")

            else:
                print(" -- Found old file without chunks. "
                      "Copying to new file with chunks...")
                old_big_file_name = dump_patch_dir + mode + "-data.h5"
                with h5py.File(old_big_file_name, "r") as old_big_file, \
                        h5py.File(big_file_name, "w-") as big_file:
                    dataset = []

                    # load old train into array
                    name_list = ["x", "y", "ID", "pos", "angle", "coords"]
                    for __i in six.moves.xrange(len(name_list)):
                        dataset += [np.asarray(old_big_file[name_list[__i]])]

                    # save train
                    big_file["time_stamp"] = np.asarray(time.localtime())

                    # allocate and write
                    for __i in six.moves.xrange(len(name_list)):
                        if name_list[__i] == "x":
                            chunk_shape = (1,) + dataset[__i].shape[1:]
                        else:
                            chunk_shape = None
                        big_file.create_dataset(
                            name_list[__i],
                            dataset[__i].shape,
                            data=dataset[__i],
                            chunks=chunk_shape,
                            maxshape=dataset[__i].shape,
                        )

                print(" -- Finished creating chunked file, removing old...")
                os.remove(old_big_file_name)

        # ----------------------------------------------------------------------
        # Copy to local tmp if necessary
        if not os.path.exists(tmp_patch_dir + mode + "-data-chunked.h5"):
            print(" -- " + mode + ": "
                  "Local dump does not exist! "
                  "Copying big dump to local drive... ")
            shutil.copy(dump_patch_dir + mode + "-data-chunked.h5",
                        tmp_patch_dir + mode + "-data-chunked.h5")
        else:
            print(" -- " + mode + ": "
                  "Local dump exists. Checking timestamp...")

            # get timestamp from nfs
            with h5py.File(dump_patch_dir + mode + "-data-chunked.h5", "r") \
                    as nfs_file:
                nfs_time = np.asarray(nfs_file["time_stamp"])

            # get timestamp from local
            with h5py.File(tmp_patch_dir + mode + "-data-chunked.h5", "r") \
                    as local_file:
                local_time = np.asarray(local_file["time_stamp"])

            # if the two files have different time stamps
            if any(nfs_time != local_time):
                print(" -- " + mode + ": "
                      "Time stamps are different! "
                      "Copying big dump to local drive... ")
                shutil.copy(dump_patch_dir + mode + "-data-chunked.h5",
                            tmp_patch_dir + mode + "-data-chunked.h5")
            else:
                print(" -- " + mode + ": "
                      "Time stamps are identical! Re-using local dump")

        # Free lock
        if os.name == "posix":
            check_lock.unlock()
            print("-- free lock --")
        elif os.name == "nt":
            check_lock.release()
            print("-- free lock --")
        else:
            pass
        # ----------------------------------------------------------------------
        # Use local copy for faster speed
        print(" -- " + mode + ": Loading from local drive... ")
        big_file_name = tmp_patch_dir + mode + "-data-chunked.h5"

        # open big_file and don"t close
        big_file = h5py.File(big_file_name, "r")

        x = big_file["x"]
        # work arround for h5py loading all things to memory
        read_batch_size = 10000
        read_batch_num = int(np.ceil(
            float(big_file["x"].shape[0]) /
            float(read_batch_size)
        ))

        # Manual, since I don't want to bother debugging the below
        # fields = ["y", "ID", "pos", "angle", "coords"]
        # for var_name in fields:
        #     # allocate data
        #     exec("{0} = np.zeros(big_file['{0}'].shape, "
        #          "dtype=big_file['{0}'].dtype)".format(var_name))
        #     # copy data in batches
        #     for idx_batch in six.moves.xrange(read_batch_num):
        #         idx_s = idx_batch * read_batch_size
        #         idx_e = (idx_batch + 1) * read_batch_size
        #         idx_e = np.minimum(idx_e, big_file["x"].shape[0])
        #         exec("{0}[idx_s:idx_e] = np.asarray(big_file['{0}'][idx_s:idx_e])"
        #              "". format(var_name))

        # Allocate
        y = np.zeros(big_file["y"].shape, dtype=big_file["y"].dtype)
        ID = np.zeros(big_file["ID"].shape, dtype=big_file["ID"].dtype)
        pos = np.zeros(big_file["pos"].shape, dtype=big_file["pos"].dtype)
        angle = np.zeros(big_file["angle"].shape,
                         dtype=big_file["angle"].dtype)
        coords = np.zeros(big_file["coords"].shape,
                          dtype=big_file["coords"].dtype)

        # Copy data in batches
        for idx_batch in six.moves.xrange(read_batch_num):
            idx_s = idx_batch * read_batch_size
            idx_e = (idx_batch + 1) * read_batch_size
            idx_e = np.minimum(idx_e, big_file["x"].shape[0])

            y[idx_s:idx_e] = np.asarray(big_file['y'][idx_s:idx_e])
            ID[idx_s:idx_e] = np.asarray(big_file['ID'][idx_s:idx_e])
            pos[idx_s:idx_e] = np.asarray(big_file['pos'][idx_s:idx_e])
            angle[idx_s:idx_e] = np.asarray(big_file['angle'][idx_s:idx_e])
            coords[idx_s:idx_e] = np.asarray(big_file['coords'][idx_s:idx_e])

        #     import pdb
        #     pdb.set_trace()

        # # Make sure data is contiguos
        # y = np.ascontiguousarray(y)
        # ID = np.ascontiguousarray(ID)
        # pos = np.ascontiguousarray(pos)
        # angle = np.ascontiguousarray(angle)
        # coords = np.ascontiguousarray(coords)

        print(" -- " + mode + ": Done... ")

        return x, y, ID, pos, angle, coords

#
# eccv.py ends here
