#!/usr/bin/env bash

bash tools/dist_train.sh configs/fyp/100_cls_dataset/byol_r50_bs128_ep200_lr2.py 2
bash tools/dist_train.sh configs/fyp/100_cls_dataset/relative_loc_r50_lr1.py 2
bash tools/dist_train.sh configs/fyp/100_cls_dataset/relative_loc_r50_lr2.py 2
bash tools/dist_train.sh configs/fyp/100_cls_dataset/rotation_pred_r50_lr5.py 2
