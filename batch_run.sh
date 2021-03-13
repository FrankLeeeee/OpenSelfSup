#!/usr/bin/env bash

bash tools/dist_train.sh configs/fyp/100_cls_dataset/rotation_pred_r50_lr1.py 2
bash tools/dist_train.sh configs/fyp/100_cls_dataset/moco_r50_v1_lr15.py 2
bash tools/dist_train.sh configs/fyp/100_cls_dataset/moco_r50_v1_lr2.py 2
bash tools/dist_train.sh configs/fyp/100_cls_dataset/moco_r50_v1_lr3.py 2


