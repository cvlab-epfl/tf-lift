# custom_types.py ---
#
# Filename: custom_types.py
# Description: Python Module for custom types
# Author: Kwang
# Maintainer:
# Created: Fri Jan 16 12:01:52 2015 (+0100)
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
#
# Copyright (C), EPFL Computer Vision Lab.
#
#

# Code:

# -------------------------------------------
# Imports
from __future__ import print_function

import hashlib
import os
import shutil
import string
from copy import deepcopy
from datetime import timedelta
from inspect import currentframe, getframeinfo

import numpy as np
from flufl.lock import Lock
from parse import parse  # ??? is parse a standard package ???


# -------------------------------------------
# Path structure
class pathConfig:
    dataset = None
    temp = None
    result = None
    debug = None

    train_data = None
    train_mask = None

    # -------------------------------------------------------------------------
    # Prefix stuff that will affect keypoint generation and pairing
    def prefix_dataset(self, param, do_sort=True):
        prefixstr = ""

        group_list = ['dataset']

        exclude_list = {}
        exclude_list['dataset'] = ['trainSetList', 'validSetList']

        for group_name in group_list:
            key_list = deepcopy(
                list(getattr(param, group_name).__dict__.keys()))
            if do_sort:
                key_list.sort()
            for key_name in key_list:
                if key_name not in exclude_list[group_name]:
                    prefixstr += str(getattr(
                        getattr(param, group_name),
                        key_name))

        prefixstr = hashlib.md5(prefixstr.encode()).hexdigest()
        prefixstr += "/"

        return prefixstr

    # --------------------------------------------------------------------------
    # Prefix stuff that will affect patch extraction
    def prefix_patch(self, param, do_sort=True):
        prefixstr = ""

        group_list = ['patch']

        exclude_list = {}
        exclude_list['patch'] = []

        for group_name in group_list:
            key_list = deepcopy(
                list(getattr(param, group_name).__dict__.keys()))
            if do_sort:
                key_list.sort()
            for key_name in key_list:
                if key_name not in exclude_list[group_name]:
                    prefixstr += str(getattr(getattr(param,
                                                     group_name), key_name))

        prefixstr = hashlib.md5(prefixstr.encode()).hexdigest()
        prefixstr += "/"

        return prefixstr

    # -------------------------------------------------------------------------
    # Prefix stuff that will affect learning outcome
    def prefix_learning(self, param, do_sort=True):
        prefixstr = ""

        group_list = ['model', 'learning']

        for group_name in group_list:
            key_list = deepcopy(
                list(getattr(param, group_name).__dict__.keys()))
            if do_sort:
                key_list.sort()
            for key_name in key_list:
                prefixstr += str(getattr(getattr(param, group_name), key_name))

        prefixstr = hashlib.md5(prefixstr.encode()).hexdigest()
        prefixstr += "/"

        return prefixstr

    def setupTrain(self, param, dataset_in):

        # Lock to prevent race condition
        if not os.path.exists(".locks"):
            os.makedirs(".locks")
        lock_file = ".locks/setup.lock"  
        if os.name == "posix":
            
            lock = Lock(lock_file)
            lock.lifetime = timedelta(days=2)
            frameinfo = getframeinfo(currentframe())
            print(">> {}/{}: waiting to obtain lock <<".format(
                    frameinfo.filename, frameinfo.lineno))
            lock.lock()
            print(">> obtained lock for posix system <<")
        elif os.name == "nt":
            import filelock
            lock = filelock.FileLock(lock_file)
            lock.acquire()
            if lock.is_locked:
                print(">> obtained lock for windows system <<")
        else:
            print("Unknown operating system, lock unavailable")
                
        

        # ---------------------------------------------------------------------
        # Base path
        # for dataset
        # self.dataset = os.getenv('PROJ_DATA_DIR', '')
        # if self.dataset == '':
        #     self.dataset = os.path.expanduser("~/Datasets")
        # self.dataset += "/" + str(dataset_in)
        self.dataset = os.path.join(
            param.data_dir, str(dataset_in)
        ).rstrip("/") + "/"
        # for temp
        # self.temp = os.getenv('PROJ_TEMP_DIR', '')
        # if self.temp == '':
        #     self.temp = os.path.expanduser("~/Temp")
        # self.temp += "/" + str(dataset_in)
        self.temp = os.path.join(
            param.temp_dir, str(dataset_in)
        ).rstrip("/") + "/"
        # for volatile temp
        # self.volatile_temp = os.getenv('PROJ_VOLTEMP_DIR', '')
        # if self.volatile_temp == '':
        #     self.volatile_temp = "/scratch/" + os.getenv('USER') + "/Temp"
        # self.volatile_temp += "/" + str(dataset_in)
        self.volatile_temp = os.path.join(
            param.scratch_dir, str(dataset_in)
        ).rstrip("/") + "/"
        # self.negdata = os.path.expanduser("~/Datasets/NegData/")  # LEGACY

        # # create dump directory if it does not exist
        # if not os.path.exists(self.temp):
        #     os.makedirs(self.temp)

        # ---------------------------------------------------------------------
        # Path for data loading

        # path for synthetic data generation
        self.train_data = self.dataset + "train/" + self.prefix_dataset(param)
        self.train_mask = (self.dataset + "train/" +
                           self.prefix_dataset(param) + "masks/")
        if not os.path.exists(self.train_data):
            # Check if the un-sorted prefix exists
            unsorted_hash_path = (self.dataset + "train/" +
                                  self.prefix_dataset(param, do_sort=False))
            if os.path.exists(unsorted_hash_path):
                os.symlink(unsorted_hash_path.rstrip("/"),
                           self.train_data.rstrip("/"))
                # shutil.copytree(unsorted_hash_path, self.train_data)
                # shutil.rmtree(unsorted_hash_path)

        # dump folder for dataset selection (sampling and etc)
        self.train_dump = self.temp + "train/" + self.prefix_dataset(param)
        if not os.path.exists(self.train_dump):
            # Check if the un-sorted prefix exists
            unsorted_hash_path = (self.temp + "train/" +
                                  self.prefix_dataset(param, do_sort=False))
            if os.path.exists(unsorted_hash_path):
                os.symlink(unsorted_hash_path.rstrip("/"),
                           self.train_dump.rstrip("/"))
                # shutil.copytree(unsorted_hash_path, self.train_dump)
                # shutil.rmtree(unsorted_hash_path)

        # dump folder for patch extraction (if necessary)
        self.patch_dump = (self.temp + "train/" + self.prefix_dataset(param) +
                           self.prefix_patch(param))
        if not os.path.exists(self.patch_dump):
            # Check if the un-sorted prefix exists
            unsorted_hash_path = (self.temp + "train/" +
                                  self.prefix_dataset(param, do_sort=False) +
                                  self.prefix_patch(param, do_sort=False))
            if os.path.exists(unsorted_hash_path):
                os.symlink(unsorted_hash_path.rstrip("/"),
                           self.patch_dump.rstrip("/"))
                # shutil.copytree(unsorted_hash_path, self.patch_dump)
                # shutil.rmtree(unsorted_hash_path)

        # volatile dump folder for patch extraction (if necessary)
        self.volatile_patch_dump = (self.volatile_temp + "train/" +
                                    self.prefix_dataset(param) +
                                    self.prefix_patch(param))
        # if not os.path.exists(self.volatile_patch_dump):
        #     # Check if the un-sorted prefix exists
        #     unsorted_hash_path = (self.volatile_temp + "train/" +
        #                           self.prefix_dataset(param, do_sort=False) +
        #                           self.prefix_patch(param, do_sort=False))
        #     os.symlink(unsorted_hash_path.rstrip("/"),
        #                self.volatile_patch_dump.rstrip("/"))
        #     # shutil.copytree(unsorted_hash_path, self.volatile_patch_dump)
        #     # shutil.rmtree(unsorted_hash_path)

        # debug info folder
        self.debug = self.dataset + "debug/" + self.prefix_dataset(param)
        if not os.path.exists(self.debug):
            # Check if the un-sorted prefix exists
            unsorted_hash_path = (self.dataset + "debug/" +
                                  self.prefix_dataset(param, do_sort=False))
            if os.path.exists(unsorted_hash_path):
                shutil.copytree(unsorted_hash_path, self.debug)
                shutil.rmtree(unsorted_hash_path)

        # # ---------------------------------------------------------------------
        # # Path for the model learning
        # resdir = os.getenv('PROJ_RES_DIR', '')
        # if resdir == '':
        #     resdir = os.path.expanduser("~/Results")
        # self.result = (resdir + "/" +
        #                self.getResPrefix(param) +
        #                self.prefix_dataset(param) +
        #                self.prefix_patch(param) +
        #                self.prefix_learning(param))
        # if not os.path.exists(self.result):
        #     # Check if the un-sorted prefix exists
        #     unsorted_hash_path = (resdir + "/" +
        #                           self.getResPrefix(param, do_sort=False) +
        #                           self.prefix_dataset(param, do_sort=False) +
        #                           self.prefix_patch(param, do_sort=False) +
        #                           self.prefix_learning(param, do_sort=False))

        #     if os.path.exists(unsorted_hash_path):
        #         shutil.copytree(unsorted_hash_path, self.result)
        #         shutil.rmtree(unsorted_hash_path)

        # # create result directory if it does not exist
        # if not os.path.exists(self.result):
        #     os.makedirs(self.result)

        if os.name == "posix":
            lock.unlock()
        elif os.name == "nt":
            lock.release()
        else:
            pass

    def getResPrefix(self, param, do_sort=True):
        trainSetList = deepcopy(list(param.dataset.trainSetList))
        if hasattr(param.dataset, "validSetList"):
            trainSetList += deepcopy(list(param.dataset.validSetList))

        if do_sort:
            trainSetList.sort()
        res_prefix = param.dataset.dataType + '/' + \
            hashlib.md5(
                "".join(trainSetList).encode()).hexdigest()

        # this is probably deprecated
        if 'prefixStr' in param.__dict__.keys():
            print('I am not deprecated!!!')
            res_prefix = param.prefixStr + "/" + res_prefix

        return res_prefix + '/'


# -------------------------------------------
# Parameter structure
class paramGroup:

    def __init__(self):
        # do nothing
        return


class paramStruct:

    def __init__(self):

        # ---------------------------------------------------------------------
        # NOTICE: everything set to None is to make it crash without config

        # ---------------------------------------------------------------------
        # Legacy
        self.legacy = paramGroup()
        # WHETHER TO GET A NEW CANNONICAL ANGLE WHETHER WE LEARN WITH SIFT
        # PRE_WARPING
        self.legacy.bRefineAngle = False
        self.legacy.bLearn4Refienment = False
        self.legacy.bUseCached = True           # Boolean for using dumped data
        # THe throw away threshold in degrees (not used if negative)
        self.legacy.fThrowAwayTh = -1
        self.legacy.fConsiderRatio = 3.0
        # If we want to learn the std to filter out bad keypoints
        self.legacy.bLearnStd = False

        self.legacy.numBin = 72
        # to mark parts to ignore when applying the shape constraint
        self.legacy.fRBFKernelTh = 0.01
        # width of the kernel in bins (kernelWidth == 0.368, 2*kernelWidth ==
        # 0.018 in shape)
        self.legacy.kernelWidth = 2.0
        # CNN_LEARNING_RATE = 0.01
        # GHH_LEARNING_RATE = 0.003
        self.legacy.hyperparamClassif = 1.0
        self.legacy.hyperparamShape = 0.01 * \
            (1.0 / (self.legacy.kernelWidth * 2 * 2 + 1))
        self.legacy.hyperparamLaplacian = 0.001

        # ---------------------------------------------------------------------
        # Parameters for SIFT descriptor
        # nNumOrientation = SIFT_ORI_HIST_BINS        # Number of orientations
        # to be used (might be deprecated!)
        self.keypoint = paramGroup()
        self.keypoint.sKpType = "SIFT"          # Descriptor type
        self.keypoint.sDescType = "SIFT"          # Descriptor type
        self.keypoint.desc_num_orientation = 72
        self.keypoint.fOverlapThresh = 40.0

        self.keypoint.fMinKpSize = 0.0       # min allowed size of a kp
        # Consider keypoints within these pixels as duplicates
        self.keypoint.fDupRange = 5.0
        self.keypoint.bNewCleanMethod = False     # for backward compatibility
        # maximum number of keypoints (-1 for unlimited)
        self.keypoint.dMaxKeypointNum = -1

        # ---------------------------------------------------------------------
        # # Paramters for patch extraction
        self.patch = paramGroup()
        # self.patch.nPatchSize = None       # Width and Height of the patch
        # self.patch.sDataType = None        # Image Data type for Patches
        # self.patch.bNormalizePatch = True  # NOrmalize single patch?
        # self.patch.bLogPolarPatch = False  # Use log-polar for patches?
        # self.patch.bPolarPatch = False     # Use log-polar for patches?
        # self.patch.bApplyWeights = None   # Whether to apply weights to input
        # # fRatioScale will be multiplied to the SIFT scale (negative uses
        # # fixed scale)
        # self.patch.fRatioScale = None
        # # use this size if smaller than this size for each kp
        # self.patch.fMinSize = None
        # self.patch.bPyramidLearn = False       # learning using pyramid patch
        # extraction

        # ---------------------------------------------------------------------
        # Parameters for synthetic images
        self.synth = paramGroup()
        # np.pi/4.0 # From -pi/4 to pi/4
        self.synth.dPerturbType = 0
        self.synth.nbAngleInPlane = 5
        self.synth.nbAngleOutPlane = 5
        self.synth.aInPlaneRot \
            = 1.0 / (self.synth.nbAngleInPlane - 1) \
            * np.arange(self.synth.nbAngleInPlane, dtype=np.float) \
            * np.pi - np.pi / 2.0  # From -pi/2 to pi/2
        self.synth.aOutPlaneRot \
            = 1.0 / (self.synth.nbAngleOutPlane - 1) \
            * np.arange(self.synth.nbAngleOutPlane, dtype=np.float) \
            * np.pi / 2.0 - np.pi / 4.0  # From -pi/4 to pi/4
        self.synth.aScaleChange = [0.5, 0.75, 1.0, 1.5, 2.0]

        self.synth.bFrontalLearn = False  # learning with frontal pair only
        self.synth.sFrontalName = 'img1.png'

        # print 'In plane rotation are: '
        # for a in aInPlaneRot:
        #     print a, ' '
        # print '\n'
        # print 'Out of plane rotation are: '
        # for a in aOutPlaneRot:
        #     print a, ' '
        # print '\n'

        # ---------------------------------------------------------------------
        # Parameters for testing
        self.PCA = paramGroup()
        self.PCA.PCAdim = None

        # ---------------------------------------------------------------------
        # Model parameters
        self.model = paramGroup()
        # Which type am I running in terms of python executable
        self.model.modelType = None
        # self.model.num_siamese = None          # number of siamese clones
        # self.model.fRatio = None               # the ratio of the WNN
        self.model.bNormalizeInput = None      # whether to normalize the input

        # ---------------------------------------------------------------------
        # Optimization parameters
        self.learning = paramGroup()
        # name of the solver (e.g. adam, adadelta)
        self.learning.optimizer = None
        self.learning.n_epochs = None      # maximum number of epochs to run
        # self.learning.min_epoch = 5        # minimum number of epochs to run
        self.learning.batch_size = None    # batch size for SGD
        # self.learning.alpha_L2 = None      # L2 regularisor weight
        # # SGD with momentum
        # self.learning.momentum = None             # momentum component
        # self.learning.learning_rate = None        # learning rate
        # # Adadelta
        # self.learning.decay = None                # ADADELTA param
        # self.learning.epsilon = None              # ADADELTA param
        # ADAM
        self.learning.lr = None
        self.learning.beta1 = None
        self.learning.beta2 = None
        self.learning.epsilon = None
        # Epoch LR decay param
        # learning rate is halved every this epoch
        self.learning.lr_half_interval = 2.0
        # reduction of learning rate after pretrain
        self.learning.pretrain_reduction = 1e-2

        # ---------------------------------------------------------------------
        # Dataset parameters
        self.dataset = paramGroup()
        self.dataset.dataType = "ECCV"

        # ---------------------------------------------------------------------
        # Validation parameters (Not used in prefix generation)
        self.validation = paramGroup()

    def loadParam(self, file_name, verbose=True):

        config_file = open(file_name, 'rb')
        if verbose:
            print("Parameters")

        # ------------------------------------------
        # Read the configuration file line by line
        while True:
            line2parse = config_file.readline().decode("utf-8")
            if verbose:
                print(line2parse, end='')

            # Quit parsing if we reach the end
            if not line2parse:
                break

            # Parse
            parse_res = parse(
                '{parse_type}: {group}.{field_name} = {read_value};{trash}',
                line2parse)

            # Skip if it is something we cannot parse
            if parse_res is not None:
                if parse_res['parse_type'] == 'ss':
                    setattr(getattr(self, parse_res['group']), parse_res[
                            'field_name'], parse_res['read_value'].split(','))
                elif parse_res['parse_type'] == 's':
                    setattr(getattr(self, parse_res['group']), parse_res[
                            'field_name'], parse_res['read_value'])
                elif parse_res['parse_type'] == 'd':
                    eval_res = eval(parse_res['read_value'])
                    if isinstance(eval_res, np.ndarray):
                        setattr(getattr(self, parse_res['group']), parse_res[
                                'field_name'], eval_res.astype(int))
                    elif isinstance(eval_res, list):
                        setattr(getattr(self, parse_res['group']), parse_res[
                                'field_name'], [int(v) for v in eval_res])
                    else:
                        setattr(getattr(self, parse_res['group']), parse_res[
                                'field_name'], int(eval_res))
                elif parse_res['parse_type'] == 'f':
                    eval_res = eval(parse_res['read_value'])
                    if isinstance(eval_res, np.ndarray):
                        setattr(getattr(self, parse_res['group']), parse_res[
                                'field_name'], eval_res.astype(float))
                    elif isinstance(eval_res, list):
                        setattr(getattr(self, parse_res['group']), parse_res[
                                'field_name'], [float(v) for v in eval_res])
                    else:
                        setattr(getattr(self, parse_res['group']), parse_res[
                                'field_name'], float(eval_res))
                elif parse_res['parse_type'] == 'sf':
                    setattr(getattr(self, parse_res['group']),
                            parse_res['field_name'],
                            [float(eval(s)) for s in
                             parse_res['read_value'].split(',')])
                elif parse_res['parse_type'] == 'b':
                    setattr(getattr(self, parse_res['group']), parse_res[
                            'field_name'], bool(int(parse_res['read_value'])))
                else:
                    if verbose:
                        print('  L-> skipped')

        # Check if we want the small range for perturbations and update it
        if self.synth.dPerturbType == 1:  # using smaller synthetics
            self.synth.nbAngleInPlane = 5
            self.synth.nbAngleOutPlane = 1
            self.synth.aInPlaneRot \
                = 1.0 / (self.synth.nbAngleInPlane - 1) \
                * np.arange(self.synth.nbAngleInPlane, dtype=np.float) \
                * np.pi / 2.0 - np.pi / 4.0  # From -pi/4 to pi/4
            self.synth.aOutPlaneRot = [0.0]
            self.synth.aScaleChange = [0.75, 1.0, 1.0 / 0.75]
        elif self.synth.dPerturbType == 2:  # using only the actual
            self.synth.nbAngleInPlane = 1
            self.synth.nbAngleOutPlane = 1
            self.synth.aInPlaneRot = [0.0]
            self.synth.aOutPlaneRot = [0.0]
            self.synth.aScaleChange = [1.0]
        elif self.synth.dPerturbType == 3:  # using only the frontal
            self.synth.nbAngleInPlane = 5
            self.synth.nbAngleOutPlane = 1
            self.synth.aInPlaneRot \
                = 1.0 / (self.synth.nbAngleInPlane - 1) \
                * np.arange(self.synth.nbAngleInPlane, dtype=np.float) \
                * np.pi / 2.0 - np.pi / 4.0  # From -pi/4 to pi/4
            self.synth.aOutPlaneRot = [0.0]
            self.synth.aScaleChange = [0.75, 1.0, 1.0 / 0.75]


# ------------------------------------------
# Data structure
# dataStruct = np.dtype([
#     ('gt_angle', np.float64),
#     ('up_angle', np.float64),
#     ('rotVec', np.float64,3)
# ])

class dataStruct:
    fSIFTAngle = 0       # SIFT angle
    fGTAngle = 0         # Ground truth angle obtained by warping the reference

    mRawPatch = []              # patch data near the keypoint region
    # patch data of a warped (according to SIFT estimation) region
    mWarpedPatch = []
    vHistogram = []        # Orientation Histogram of the selected patch
    vWarpedHistogram = []  # Warped Orientation Histogram of the selected patch

    mRefRawPatch = []           # patch data near the keypoint region
    # patch data of a warped (according to SIFT estimation) region
    mRefWarpedPatch = []

    # patch data of a warped (according to GT estimation) region
    mGTWarpedPatch = []

    vRefHistogram = []       # Orientation Histogram of the selected patch
    vRefWarpedHistogram = []  # Warped Orient. Histogram of the selected patch

    # descriptor computed for all angles (the name is misleading here!)
    mOrientedSIFTdesc = []
    vSIFTdesc = []              # descriptor computed with SIFT angle
    vGTdesc = []                # descriptor computed with GT angle

    dLabel = 0     # Label of the data
    fLabelStd = 0  # Std of the keypoints having this label in the training set
    pt = []        # coordinates of the keypoint

    sFileName = []           # Original File name which this data was extracted

    # ------------------------------------------------------------------------
    # Fro Brown data
    refID = None
    kpID = None
    interest_x = None
    interest_y = None
    interest_orientation = None
    interest_scale = None
    # mOrigPatch = None


class data2DrawStruct:
    # orig_keypoint = cv2.KeyPoint()        # Opencv Keypoint
    # updt_keypoint = cv2.KeyPoint()        # Opencv Keypoints with updated
    # angle (recomputed)
    gt_angle = 0
    up_angle = 0
    # rotation Vector used to synthesize the image and keypoint
    rotVec = np.zeros((3))


#
# custom_types.py ends here
