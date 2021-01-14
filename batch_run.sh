#!/usr/bin/env bash

bash ./benchmarks/dist_test_svm_epoch.sh ./configs/fyp/100_cls_dataset/contrastive_odc_r50_v12.py  200 "feat3 feat4 feat5" 2
bash tools/dist_train.sh configs/fyp/20_cls_dataset/contrastive_odc_r50_v13.py 2
bash ./benchmarks/dist_test_svm_epoch.sh ./configs/fyp/20_cls_dataset/contrastive_odc_r50_v13.py  200 "feat3 feat4 feat5" 2

