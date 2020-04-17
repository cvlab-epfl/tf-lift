# TF-LIFT: Tensorflow Implmentation for Learned Invariant Feature Transform #

## Basic Usage ##

### Setting the environment ###

Python Version : 3
OpenCV Version : 3

The base dependencies in Python3 are :

```
pip install numpy h5py tensorflow=1.4.0 tensorflow-gpu=1.4.0

```
We will later provide a requirements.txt for you to use with `pip`.

If working with an Anaconda environment on Windows, you can create the environment from the `tf-lift-env.yml` file or the `tfliftspec.txt` file.

Also, you need to setup your work directories. Edit the `config.py` for a
convenient default argument setting. See help for more information on what the
configurations do.

### Pre-Processing ###

1. Download a dataset from: http://www.cs.cornell.edu/projects/1dsfm/. The LIFT paper uses the Piccadilly and Roman Forum datasets separately.
2. Extract it somewhere.
3. Call: `VisualSFM.exe` pointed at the directory which contains the extracted images.
	- Calling the VSFM GUI ensures CUDA usage.
4. Call: `python convert_asift_vsfm_to_h5.py` (help will explain args)


### Training ###

`main.py` is the entry point for all your needs. Simply define the task you
want to do, the where to save the results (logs) and the training subtask you
want to perform.

For example, to train the keypoint descriptor, then orientation, then detector, and then to run the joint trainer:
```
python main.py --task=train --subtask=desc
python main.py --task=train --subtask=ori
python main.py --task=train --subtask=kp
python main.py --task=train --subtask=joint
```

Note: this will save the logs at `logs/main.py---task=train---subtask=desc`. If
you don't want this behavior, you can also add `--logdir=logs/test` in the
command line argument, for example.

To train with specific checkpoints for performance evaluation, you can run the `run_iteration_blocks.py` script. This will checkpoint the training process at values specified in the script, using the values from each checkpoint to seed the following runs.

### Testing ###

To run test passes on images, call the `run_test_pass_on_image.py` script. For example, the following
command will run the entire pipeline for `image1.jpg`, using the model at `logs/test`.

```
run_test_pass_on_image.py -i image1.jpg -m logs/test
run_test_pass_on_image.py -i image2.jpg -m logs/test
```

Note: when trying to load the model, it will always look for the `joint`
trained model first, and fall back to the subtask it is trying to test for.

Then to evaluate the results, call the `match_network_outputs.py` script:
```
match_network_outputs.py image1 image1.h5 image2 image2.h5
```

## More notes on training ##

### Saving the network ###

When training, the network is automatically saved in the `logdir`. If you don't
set this manually, it defaults to
`logs/{concatenation-of-all-arguments}`. The things that are saved are:

* Tensorflow checkpoint
* Tensorflow graph metadata
* `mean` and `std` used for input data normalization
* best validation loss
* best validation iteration

All these are loaded back when we want to continue.

### Loading the network ###

On all runs, the framework automatically resumes from where it left. In other
words, it will **always** try to load network weights and resume. If the
framework cannot find the expected weights, it will just tell you that it could
not find weights in the expected locations, and **will try to go on its merry
way**. Note that this is something that you want to keep in mind. For example,
if you run the subtask `ori`, with a typo in `logdir` pointing you to a
directory without the pretrained descriptor weights, the framework will simply
try to learn the orientation estimator **with random descriptors**. This is
intended, as this might be something that you actually want to try.

Network loading is performed in the following order, **overwriting** the previously
loaded weights:

1. Loads the pretrained weights, in the **old framework** format, from
   directories defined in `pretrained_{subtask}` in the configuration. **This
   feature is deprecated and should not be used**

2. Loads the pretrained weights, in the **new framework** format, from
   directories defined in `pretrained_{subtask}` in the configuration.

2. Loads the weights from the `logdir`, which is either automatically determined
   by the command line arguments, or can be given manually.
   
## Differences from the original version ##

### Batch normalization ###

In the original version, we did not apply batch normalization. In the new
version, bach normalization is applied to all layers. This significantly
speeds-up the learning process, and makes learning stable. This also eliminates
the need for us to normalize the dataset when training, and we can instead
simply put the data range in a reasonable range, say -1 to 1 and be done with
it. Note that since we do this, we also perform batch normalization on the
input.

### L2-pooling and Spatial subtractive normalization ###

We found that these layers can be replaced with normal relus and spatial
pooling without significant difference. They are removed.

## Pretrained models ##

We provide new models trained on the `Piccadilly` set from the ECCV paper.
Note that they have been trained from scratch with the new framework (as
opposed to [the theano-based framework we used at the time of the ECCV
submission](https://github.com/cvlab-epfl/LIFT)), so there are some changes in
the architecture and training procedure. Performance should be about on par.

The files can be downloaded here:
* [Models without rotation augmentation](http://webhome.cs.uvic.ca/~kyi/files/2018/tflift/release-no-aug.tar.gz) (run with --use_batch_norm=False --mean_std_type=dataset)
* [Models with rotation augmentation](http://webhome.cs.uvic.ca/~kyi/files/2018/tflift/release-aug.tar.gz) (run with --use_batch_norm=False --mean_std_type=hardcoded)

The models trained without rotation augmentation perform better on matching
problems where the images are generally upright. For data with random
rotations, use the models trained with rotation augmentation.
