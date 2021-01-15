#!/usr/bin/env bash

bash ./benchmarks/dist_test_svm_epoch.sh ./configs/fyp/100_cls_dataset/contrastive_odc_r50_v14.py  200 "feat4 feat5" 2

bash ./benchmarks/dist_test_svm_epoch.sh ./configs/fyp/100_cls_dataset/contrastive_odc_r50_v14.py  150 "feat4 feat5" 2
bash ./benchmarks/dist_test_svm_epoch.sh ./configs/fyp/100_cls_dataset/contrastive_odc_r50_v14.py  100 "feat4 feat5" 2


