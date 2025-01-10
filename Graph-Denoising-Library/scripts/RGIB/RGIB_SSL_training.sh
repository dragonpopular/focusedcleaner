#!/bin/bash

python -u run.py \
    --model RGIB \
    --debug \
    --gnn_model GCN \
    --num_gnn_layers 4 \
    --dataset Cora \
    --noise_ratio 0.2 \
    --scheduler linear \
    --scheduler_param 1.0 \
    --task_name linkPrediction \



# 参数列表 可配置在run的option中
# --model RGIB --debug --gnn_model GCN --num_gnn_layers 4 --dataset Cora --noise_ratio 0.2 --scheduler linear --scheduler_param 1.0 --task_name linkPrediction