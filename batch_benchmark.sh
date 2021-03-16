#!/usr/bin/env bash

for CONFIG in "moco_r50_v1_lr3"  "moco_r50_v2_lr3"
do
    bash ./benchmarks/dist_test_svm_epoch.sh "./configs/fyp/100_cls_dataset/${CONFIG}.py"  200 "feat4 feat5" 1
    python tools/extract_backbone_weights.py "./work_dirs/fyp/100_cls_dataset/${CONFIG}/epoch_200.pth"  "./work_dirs/extracted_weights/${CONFIG}_200.pth"
    bash benchmarks/dist_train_linear.sh configs/benchmarks/fyp/imagenet_linear_classification/r50_last.py "./work_dirs/extracted_weights/${CONFIG}_200.pth"
done


# for CONFIG in "rotation_pred_r50_lr1"
# do
#     bash ./benchmarks/dist_test_svm_epoch.sh "./configs/fyp/100_cls_dataset/${CONFIG}.py"  70 "feat4 feat5" 1
#     python tools/extract_backbone_weights.py "./work_dirs/fyp/100_cls_dataset/${CONFIG}/epoch_70.pth"  "./work_dirs/extracted_weights/${CONFIG}_70.pth"
#     bash benchmarks/dist_train_linear.sh configs/benchmarks/fyp/imagenet_linear_classification/r50_last.py "./work_dirs/extracted_weights/${CONFIG}_70.pth"
# done
