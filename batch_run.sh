#!/usr/bin/env bash

bash ./benchmarks/dist_test_svm_epoch.sh ./configs/fyp/100_cls_dataset/odc_r50_bs128_v1_gpu1.py.py  200 "feat4 feat5" 1
bash ./benchmarks/dist_test_svm_epoch.sh ./configs/fyp/100_cls_dataset/simclr_r50_bs128_ep200.py  200 "feat4 feat5" 1


