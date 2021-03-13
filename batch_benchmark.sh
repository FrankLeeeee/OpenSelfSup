#!/usr/bin/env bash

for CONFIG in "byol_r50_bs128_ep200_lr2"
do
    bash ./benchmarks/dist_test_svm_epoch.sh "./configs/fyp/100_cls_dataset/${CONFIG}.py"  200 "feat4 feat5" 1
    python tools/extract_backbone_weights.py "./work_dirs/fyp/100_cls_dataset/${CONFIG}/epoch_200.pth"  "./work_dirs/extracted_weights/${CONFIG}_200.pth"
    bash benchmarks/dist_train_linear.sh configs/benchmarks/fyp/imagenet_linear_classification/r50_last.py "./work_dirs/extracted_weights/${CONFIG}_200.pth"
done


for CONFIG in "relative_loc_r50_lr1" "relative_loc_r50_lr2" "rotation_pred_r50_lr5"
do
    bash ./benchmarks/dist_test_svm_epoch.sh "./configs/fyp/100_cls_dataset/${CONFIG}.py"  70 "feat4 feat5" 1
    python tools/extract_backbone_weights.py "./work_dirs/fyp/100_cls_dataset/${CONFIG}/epoch_70.pth"  "./work_dirs/extracted_weights/${CONFIG}_70.pth"
    bash benchmarks/dist_train_linear.sh configs/benchmarks/fyp/imagenet_linear_classification/r50_last.py "./work_dirs/extracted_weights/${CONFIG}_70.pth"
done
