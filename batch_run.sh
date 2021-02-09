#!/usr/bin/env bash

bash ./benchmarks/dist_test_svm_epoch.sh ./configs/fyp/100_cls_dataset/contrastive_odc_r50_v24_bs128.py  200 "feat4 feat5" 1
python tools/extract_backbone_weights.py ./work_dirs/fyp/100_cls_dataset/contrastive_odc_r50_v26_ncluster10_mincluster100/epoch_200.pth ./work_dirs/extracted_weights/contrastive_odc_r50_v26_ncluster10_mincluster100_epoch_200.pth
bash benchmarks/dist_train_linear.sh configs/benchmarks/fyp/imagenet_linear_classification/r50_last.py ./work_dirs/extracted_weights/contrastive_odc_r50_v26_ncluster10_mincluster100_epoch_200.pth



