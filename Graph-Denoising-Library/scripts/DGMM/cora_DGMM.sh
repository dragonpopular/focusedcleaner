#!/bin/bash

python -u run.py \
       --model DGMM  \
       --debug \
       --dataset Cora \
       --tau 0.6   \
       --beta 0.3 \
       --temperature 2 \
       --hidden 16 \
       --atk nettack\
       --dataset cora  \
       --dataBy classic \
       --ptb_rate 0.05\
       --n_ptb 1 \
       --san_rate 0.1 \




