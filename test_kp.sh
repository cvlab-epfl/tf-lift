#!/bin/bash
CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=10 python main.py --test_img_file=./test/fountain.png --test_out_file=./test/fountain_kp.txt --task=test --subtask=kp --logdir="logs/main.py"
