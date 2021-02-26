#!/usr/bin/env bash

bash tools/dist_train.sh configs/fyp/100_cls_dataset/contrastive_odc_r50_v24_bs128_lr2_classes200_mincluster32.py 2
bash tools/dist_train.sh configs/fyp/100_cls_dataset/contrastive_odc_r50_v24_bs128_lr2_classes800_mincluster32.py 2
bash tools/dist_train.sh configs/fyp/100_cls_dataset/contrastive_odc_r50_v24_bs128_lr2_classes1600_mincluster32.py 2

