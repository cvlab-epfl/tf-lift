# trainer.py ---
#
# Filename: trainer.py
# Description:
# Author: Kwang Moo Yi, Lin Chen
# Maintainer: Kwang Moo Yi
# Created: ???
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
# Tue 27 Jun 15:04:34 CEST 2017, Kwang Moo Yi
#
# - Removed comments.
# - Renamed Dataset import
#
#

# Code:


import os

import numpy as np
import tensorflow as tf
from tqdm import trange

from datasets.lift import Dataset
from networks.lift import Network
from six.moves import xrange
from utils import get_hard_batch, restore_network, save_network


class Trainer(object):
    """The Trainer Class

    LATER: Remove all unecessary "dictionarization"

    """

    def __init__(self, config, rng):
        self.config = config
        self.rng = rng

        # Open a tensorflow session. I like keeping things simple, so I don't
        # use a supervisor. I'm just going to do everything manually. I also
        # will just allow the gpu memory to grow
        tfconfig = tf.ConfigProto()
        if self.config.usage > 0:
            tfconfig.gpu_options.allow_growth = False
            tfconfig.gpu_options.per_process_gpu_memory_fraction = \
                self.config.usage
        else:
            tfconfig.gpu_options.allow_growth = True
        self.sess = tf.Session(config=tfconfig)

        # Create the dataset instance
        self.dataset = Dataset(self.config, rng)
        self.network = Network(self.sess, self.config, self.dataset)
        # Make individual saver instances and summary writers for each module
        self.saver = {}
        self.summary_writer = {}
        self.best_val_loss = {}
        self.best_step = {}
        # Saver (only if there are params)
        for _key in self.network.allparams:
            if len(self.network.allparams[_key]) > 0:
                with tf.variable_scope("saver-{}".format(_key)):
                    self.saver[_key] = tf.train.Saver(
                        self.network.allparams[_key])
        # Summary Writer
        # Disable this, it's too big
        self.summary_writer[self.config.subtask] = tf.summary.FileWriter(
            os.path.join(self.config.logdir, self.config.subtask),
            # graph=self.sess.graph
        )
        # validation loss
        self.best_val_loss[self.config.subtask] = np.inf
        # step for each module
        self.best_step[self.config.subtask] = 0

        # We have everything ready. We finalize and initialie the network here.
        self.sess.run(tf.global_variables_initializer())

        # Enable augmentations and/or force the use of the augmented set
        self.use_aug_rot = 0
        if self.config.augment_rotations:
            self.use_aug_rot = 1
        elif self.config.use_augmented_set:
            self.use_aug_rot = -1

    def run(self):
        # For each module, check we have pre-trained modules and load them
        print("-------------------------------------------------")
        print(" Looking for previous results ")
        print("-------------------------------------------------")
        for _key in ["kp", "ori", "desc", "joint"]:
            restore_network(self, _key)

        print("-------------------------------------------------")
        print(" Training ")
        print("-------------------------------------------------")

        # Sanity check: no BN when training keypoints
        # if self.config.substask == "kp" or (
        #     self.config.subtask == "joint" and
        #         "kp" in self.config.finetune):
        #     raise RuntimeError("Training keypoints with Batch Normalization"
        #                        "enabled is not allowed.")

        subtask = self.config.subtask
        batch_size = self.config.batch_size
        for step in trange(int(self.best_step[subtask]),
                           int(self.config.max_step),
                           desc="Subtask = {}".format(subtask),
                           ncols=self.config.tqdm_width):
            # ----------------------------------------
            # Forward pass: Note that we only compute the loss in the forward
            # pass. We don't do summary writing or saving
            fw_data = []
            fw_loss = []
            batches = self.hardmine_scheduler(self.config, step)
            for num_cur in batches:
                cur_data = self.dataset.next_batch(
                    task="train",
                    subtask=subtask,
                    batch_size=num_cur,
                    aug_rot=self.use_aug_rot)
                cur_loss = self.network.forward(subtask, cur_data)
                # Sanity check
                if min(cur_loss) < 0:
                    raise RuntimeError('Negative loss while mining?')
                # Data may contain empty (zero-value) samples: set loss to zero
                if num_cur < batch_size:
                    cur_loss[num_cur - batch_size:] = 0
                fw_data.append(cur_data)
                fw_loss.append(cur_loss)
            # Fill a single batch with hardest
            if len(batches) > 1:
                cur_data = get_hard_batch(fw_loss, fw_data)
            # ----------------------------------------
            # Backward pass: Note that the backward pass returns summary only
            # when it is asked. Also, we manually keep note of step here, and
            # not use the tensorflow version. This is to simplify the migration
            # to another framework, if needed.
            do_validation = step % self.config.validation_interval == 0
            cur_summary = self.network.backward(
                subtask, cur_data, provide_summary=do_validation)

            if do_validation and cur_summary is not None:
                # Make sure we have the summary data
                assert cur_summary is not None
                # Write training summary
                self.summary_writer[subtask].add_summary(cur_summary, step)
                # Do multiple rounds of validation
                cur_val_loss = np.zeros(self.config.validation_rounds)
                for _val_round in xrange(self.config.validation_rounds):
                    # Fetch validation data
                    cur_data = self.dataset.next_batch(
                        task="valid",
                        subtask=subtask,
                        batch_size=batch_size,
                        aug_rot=self.use_aug_rot)
                    # Perform validation of the model using validation data
                    cur_val_loss[_val_round] = self.network.validate(
                        subtask, cur_data)
                cur_val_loss = np.mean(cur_val_loss)
                # Inject validation result to summary
                summaries = [
                    tf.Summary.Value(
                        tag="validation/err-{}".format(subtask),
                        simple_value=cur_val_loss,
                    )
                ]
                self.summary_writer[subtask].add_summary(
                    tf.Summary(value=summaries), step)
                # Flush the writer
                self.summary_writer[subtask].flush()

                # TODO: Repeat without augmentation if necessary
                # ...

                if cur_val_loss < self.best_val_loss[subtask]:
                    self.best_val_loss[subtask] = cur_val_loss
                    self.best_step[subtask] = step
                    save_network(self, subtask)

    def hardmine_scheduler(self, config, step, recursion=True):
        """The hard mining scheduler.

        Modes ("--mining-sched"):
            "none": no mining.
            "step": increase one batch at a time.
            "smooth": increase one sample at a time, filling the rest of the
            batch with zeros if necessary.

        Returns a list with the number of samples for every batch.
        """

        sched = config.mining_sched
        if sched == 'none':
            return [config.batch_size]
        elif sched not in ['step', 'smooth']:
            raise RuntimeError('Unknown scheduler')

        # Nothing to do if mining_step is not defined
        if config.mining_step <= 0:
            return [config.batch_size]

        max_batches = config.mining_ceil if config.mining_ceil > 0 else 32
        num_samples = int(round(config.batch_size *
                          (config.mining_base + step / config.mining_step)))
        if num_samples > max_batches * config.batch_size:
            # Limit has been reached
            batches = [config.batch_size] * max_batches
        else:
            batches = [config.batch_size] * int(num_samples // config.batch_size)
            # Do the remainder on the last batch
            remainder = num_samples % config.batch_size
            if remainder > 0:
                # 'smooth': add remainder to the last batch
                if sched == 'smooth':
                    batches[-1] += remainder
                # 'step': add a full batch when the remainder goes above 50%
                elif sched == 'step' and remainder >= config.batch_size / 2:
                    batches += [config.batch_size]

        # Feedback
        if recursion and step > 0:
            prev = self.hardmine_scheduler(config, step - 1, recursion=False)
            if sum(prev) < sum(batches):
                print(('\n[{}] Mining: increasing number of samples: ' +
                       '{} -> {} (batches: {} -> {}, last batch size: {})').format(
                    config.subtask,
                    sum(prev),
                    sum(batches),
                    len(prev),
                    len(batches),
                    batches[-1]))

        return batches

#
# trainer.py ends here
