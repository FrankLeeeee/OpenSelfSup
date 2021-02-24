#!/usr/bin/env bash

for EPOCH in 40 80 120 160 200
do
    python tools/extract_backbone_weights.py ./work_dirs/fyp/100_cls_dataset/simclr_r50_bs128_ep200_lr2_gpu1/epoch_${EPOCH}.pth  ./work_dirs/extracted_weights/simclr_r50_bs128_ep200_lr2_gpu1_epoch_${EPOCH}.pth
    bash benchmarks/dist_train_linear.sh configs/benchmarks/fyp/imagenet_linear_classification/r50_last.py ./work_dirs/extracted_weights/simclr_r50_bs128_ep200_lr2_gpu1_epoch_${EPOCH}.pth
done


