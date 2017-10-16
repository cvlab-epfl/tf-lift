#!/bin/bash
CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=10 python main.py --test_img_file=./test/fountain.png --test_kp_file=./test/fountain_kp.txt --test_out_file=./test/fountain_ori.txt --task=test --subtask=ori --logdir="logs/main.py"
