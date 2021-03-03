#!/usr/bin/env bash

bash tools/dist_train.sh configs/fyp/100_cls_dataset/contrastive_odc_r50_v24_bs128_lr2_classes400_mincluster32_trivial_test.py 2
bash tools/dist_train.sh configs/fyp/100_cls_dataset/odc_r50_v1_trivial_test.py 2
