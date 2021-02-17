#!/usr/bin/env bash

bash tools/dist_train.sh configs/fyp/100_cls_dataset/contrastive_odc_r50_v7_2_bs128_lr2_classes400_mincluster_32.py 2
bash tools/dist_train.sh configs/fyp/100_cls_dataset/contrastive_odc_r50_v7_2_bs128_lr2_classes250_mincluster_50.py 2
bash tools/dist_train.sh configs/fyp/100_cls_dataset/contrastive_odc_r50_v12_2_bs128_lr2_classes250_mincluster50.py 2
bash tools/dist_train.sh configs/fyp/100_cls_dataset/contrastive_odc_r50_v12_2_bs128_lr2_classes500_mincluster20.py 2



