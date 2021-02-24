#!/usr/bin/env bash

bash tools/dist_train.sh configs/fyp/100_cls_dataset/contrastive_odc_r50_v24_bs128_lr2_classes100_mincluster64.py 2
bash tools/dist_train.sh configs/fyp/100_cls_dataset/contrastive_odc_r50_v24_bs128_lr2_classes100_mincluster256.py 2
bash tools/dist_train.sh configs/fyp/100_cls_dataset/contrastive_odc_r50_v24_bs128_lr2_classes100_mincluster512.py 2


