# config.py ---
#
# Filename: config.py
# Description: Configuration module.
# Author: Kwang Moo Yi
# Maintainer: Kwang Moo Yi
# Created: Wed Jun 28 13:08:23 2017 (+0200)
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
# Adapted from original code at
# https://github.com/carpedm20/simulated-unsupervised-tensorflow
#
#

# Change Log:
#
#
#

# Code:


import argparse
import getpass
import json
import os
from utils.config import get_patch_size


def str2bool(v):
    return v.lower() in ("true", "1")


arg_lists = []
parser = argparse.ArgumentParser()
username = getpass.getuser()


def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg


# ----------------------------------------
# Network
net_arg = add_argument_group("Network")
net_arg.add_argument("--kp_input_size", type=int, default=48, help="")
net_arg.add_argument("--kp_filter_size", type=int, default=25, help="")
net_arg.add_argument("--kp2_num_layers", type=int, default=5, help="")
net_arg.add_argument("--kp_base_scale", type=float, default=2.0, help="")
net_arg.add_argument("--kp_com_strength", type=float, default=10.0, help="")
net_arg.add_argument("--kp_num_fc_units", type=int, default=0, help="For module lift_kp_noghh")
# net_arg.add_argument("--ori_input_size", type=int, default=28, help="")
net_arg.add_argument("--ori_input_size", type=int, default=64, help="")
net_arg.add_argument("--desc_input_size", type=int, default=64, help="")
net_arg.add_argument("--desc_support_ratio", type=float, default=6.0, help="")

# Module selection
net_arg.add_argument("--module_kp", type=str, default="lift_kp", help="")
net_arg.add_argument("--module_ori", type=str, default="lift_ori", help="")
net_arg.add_argument("--module_desc", type=str, default="lift_desc", help="")

# Batch-norm on-off
net_arg.add_argument("--use_input_batch_norm", type=str2bool, default=False, help="")
net_arg.add_argument("--use_batch_norm", type=str2bool, default=True, help="")

# Data compatibility option
net_arg.add_argument("--old_data_compat", type=str2bool, default=False, help="Use hard-mined, non-augmented set")

# Orientation detector options
net_arg.add_argument("--use_augmented_set", type=str2bool, default=False, help="Use/extract the dataset for augmented rotations")
net_arg.add_argument("--augment_rotations", type=str2bool, default=False, help="")
net_arg.add_argument("--use_dropout_ori", type=bool, default=False, help="")
net_arg.add_argument("--ori_activation", type=str, default="ghh", choices=["ghh", "tanh"], help="")

# Descriptor options
net_arg.add_argument("--desc_activ", type=str, default="relu", help="Descriptor activation")
net_arg.add_argument("--desc_pool", type=str, default="avg_pool", help="Descriptor pooling")
net_arg.add_argument("--use_subtractive_norm", type=str2bool, default=False, help="Descriptor subtractive normalization")
net_arg.add_argument("--use_hardest_anchor", type=str2bool, default=True, help="Use hardest anchor")
net_arg.add_argument("--use_triplet_loss", type=str2bool, default=True, help="Triplet loss")
net_arg.add_argument("--triplet_loss_margin", type=float, default=5, help="Triplet loss margin")

# Use old mean/std
# net_arg.add_argument("--use_old_mean_std", type=str2bool, default=False, help="")
net_arg.add_argument("--mean_std_type", type=str, default="hardcoded",
                     choices=["hardcoded", "old", "dataset", "sample", "batch"], help="")

# ----------------------------------------
# Loss Function
loss_arg = add_argument_group("Loss")
loss_arg.add_argument("--alpha_overlap", type=float, default=1e0, help="")
loss_arg.add_argument("--alpha_classification", type=float, default=1e-8,
                      help="")
loss_arg.add_argument("--alpha_margin", type=float, default=4.0, help="")
# for joint training
loss_arg.add_argument("--alpha_kp", type=float, default=1e0, help="")
loss_arg.add_argument("--alpha_desc", type=float, default=1e0, help="")
loss_arg.add_argument("--kp_scoremap_softmax_strength",
                      type=float, default=10.0, help="")

# ----------------------------------------
# Data
data_arg = add_argument_group("Data")
data_arg.add_argument("--use_local", type=str2bool, default=True, help="")
data_arg.add_argument("--nchannel", type=int, default=1, help="")
data_arg.add_argument("--data_type", type=str, default="eccv2016", help="")
data_arg.add_argument("--data_name", type=str, default="piccadilly", help="")
data_arg.add_argument(
    "--data_dir", type=str, help=(
        "The directory containing the dataset. "
        "Will look for {data_dir}/{data_name}"
    ), default="/cvlabdata2/home/{}/Datasets/".format(username),
)
data_arg.add_argument(
    "--temp_dir", type=str, help=(
        "The temporary directory where data related cache will be stored."
    ), default="/cvlabdata2/home/{}/Temp/".format(username),
)
data_arg.add_argument(
    "--scratch_dir", type=str, help=(
        "The temporary directory that will be used as cache."
        "We have this since the large data is typically stored in a "
        "network share"
    ), default="/scratch/{}/Temp/".format(username),
)
data_arg.add_argument(
    "--pair_dir", type=str, help=(
        "Creating pairs are time consuming. "
        "We store the pair in this directory. "
        "This behavior might be removed in the future. "
    ), default="./pairs",
)
data_arg.add_argument("--regen_pairs", type=str2bool, default=True, help="")

# ----------------------------------------
# Task
task_arg = add_argument_group("Task")
task_arg.add_argument("--task", type=str, default="train",
                      choices=["train", "test"],
                      help="")
task_arg.add_argument("--subtask", type=str, default="desc",
                      choices=["kp", "ori", "desc", "joint"],
                      help="")
task_arg.add_argument("--logdir", type=str, default="", help="")
task_arg.add_argument("--finetune", type=str, default="kp", help="e.g. 'kp+ori+desc'")

# ----------------------------------------
# Training
train_arg = add_argument_group("Train")
train_arg.add_argument("--random_seed", type=int, default=1234, help="")
train_arg.add_argument("--batch_size", type=int, default=128, help="")
train_arg.add_argument("--pair_interval", type=int, default=1, help="")
train_arg.add_argument("--pair_use_cache", type=str2bool,
                       default=True, help="")
train_arg.add_argument("--max_step", type=int, default=1e8, help="")
train_arg.add_argument("--optimizer", type=str, default="adam",
                       choices=["adam", "rmsprop", "sgd"],
                       help="")
train_arg.add_argument("--learning_rate", type=float, default=1e-3, help="")
train_arg.add_argument("--max_grad_norm", type=float, default=-1.0, help="")
train_arg.add_argument("--check_numerics", type=str2bool,
                       default=True, help="")
train_arg.add_argument("--tqdm_width", type=int, default=79, help="")
train_arg.add_argument("--mining_sched", type=str, default="none",
                       help="Scheduler: 'none', 'step', 'smooth'")
train_arg.add_argument("--mining_base", type=int, default=1,
                       help="Starting number of batches")
train_arg.add_argument("--mining_step", type=int, default=0,
                       help="Add one batch every these many (0 to disable)")
train_arg.add_argument("--mining_ceil", type=int, default=0,
                       help="Max number of batches (0 to disable)")

# Pretrain information to force in if needed
train_arg.add_argument("--pretrained_kp", type=str, default="", help="")
train_arg.add_argument("--pretrained_ori", type=str, default="", help="")
train_arg.add_argument("--pretrained_desc", type=str, default="", help="")
train_arg.add_argument("--pretrained_joint", type=str, default="", help="")

# ----------------------------------------
# Validation
valid_arg = add_argument_group("Validation")
valid_arg.add_argument("--validation_interval", type=int, default=1e3, help="")
valid_arg.add_argument("--validation_rounds", type=int, default=100, help="")
valid_arg.add_argument("--neg_per_pos", type=float, default=100.0, help="")
valid_arg.add_argument("--valid_method", type=str, default="desc", help="")

# ----------------------------------------
# Test
test_arg = add_argument_group("Test")
test_arg.add_argument("--test_img_file", type=str, default="", help="")
test_arg.add_argument("--test_kp_file", type=str, default="", help="")
test_arg.add_argument("--test_out_file", type=str, default="", help="")
test_arg.add_argument("--test_num_keypoint", type=int, default=1000, help="")
test_arg.add_argument("--test_scl_intv", type=int, default=4, help="")
test_arg.add_argument("--test_min_scale_log2", type=int, default=1, help="")
test_arg.add_argument("--test_max_scale_log2", type=int, default=4, help="")
test_arg.add_argument("--test_kp_use_tensorflow",
                      type=str2bool, default=True, help="")
test_arg.add_argument("--test_nearby_ratio", type=float, default=1.0, help="")
test_arg.add_argument("--test_nms_intv", type=int, default=2, help="")
test_arg.add_argument("--test_edge_th", type=float, default=10.0, help="")

train_arg = add_argument_group("Misc")
loss_arg.add_argument("--usage", type=float, default=0.96, help="Force GPU memory usage")


def get_config(argv):
    config, unparsed = parser.parse_known_args()

    # Sanity checks
    if config.augment_rotations and not config.use_augmented_set:
        config.use_augmented_set = True
        print('-- Forcing use_augmented_set = True')

    if config.augment_rotations and config.subtask is "desc":
        raise RuntimeError("Rotation augmentation is incompatible "
                           "with descriptor training.")

    if config.old_data_compat and (
            config.use_augmented_set or config.augment_rotations):
        raise RuntimeError("Options incompatible with legacy data generation.")

    if config.subtask == 'joint':
        what = config.finetune.split('+')
        if ("kp" not in what) and ("ori" not in what) and \
                ("desc" not in what):
            raise RuntimeError("Nothing to finetune? Check --finetune")


    # Create the prefix automatically from run command
    if config.logdir == "":
        config.logdir = "-".join(argv)
        config.logdir = os.path.join("logs",
                                     config.logdir.replace("main.py", ""))

    return config, unparsed


def save_config(model_dir, config):
    param_path = os.path.join(model_dir, config.subtask, "params.json")

    print("[*] MODEL dir: %s" % model_dir)
    print("[*] PARAM path: %s" % param_path)

    with open(param_path, "w") as fp:
        json.dump(config.__dict__, fp, indent=4, sort_keys=True)

#
# config.py ends here
