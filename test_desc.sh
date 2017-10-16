#!/bin/bash
CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=10 python main.py --test_img_file=./test/fountain.png --test_kp_file=./test/fountain_ori.txt --test_out_file=./test/fountain_desc.h5 --task=test --subtask=desc --logdir="logs/main.py"
