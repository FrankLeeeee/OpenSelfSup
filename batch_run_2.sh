#!/usr/bin/env bash

bash tools/dist_train.sh configs/fyp/100_cls_dataset/moco_r50_v2_lr2_gpu1.py 1
bash tools/dist_train.sh configs/fyp/100_cls_dataset/moco_r50_v2_lr3_gpu1.py 1

