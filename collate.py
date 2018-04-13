from os import listdir
from os.path import isfile, isdir, join
from argparse import ArgumentParser
import tensorflow as tf
from utils.dump import loadh5


def look_for_checkpoints(folder, task):
    dirs = [f
            for f in listdir(folder)
            if isdir(join(folder, f))]

    for d in dirs:
        # print(join(folder, d))
        if d == task or (d in ['kp', 'ori', 'desc', 'joint'] and task == 'all'):
            cp = tf.train.latest_checkpoint(join(folder, d))
            if cp is not None:
                # Load best validation result
                try:
                    r = loadh5(join(folder, d, 'best_val_loss.h5'))[d]
                    s = loadh5(join(folder, d, 'step.h5'))[d]
                    print('{0:s} -> {1:.05f} [{2:d}]'.format(join(folder, d), r, s))
                except:
                    print("Could not open '{}' for '{}'".format(d, folder))
        else:
            look_for_checkpoints(join(folder, d), task)


parser = ArgumentParser()
parser.add_argument('--logdir', type=str, required=True)
parser.add_argument('--task', type=str, default='all', help='kp, ori, desc, joint, all')
params = parser.parse_args()

print()
print('*** Logdir: "{}" (task: {}) ***'.format(params.logdir, params.task))
look_for_checkpoints(params.logdir, params.task)
print()
