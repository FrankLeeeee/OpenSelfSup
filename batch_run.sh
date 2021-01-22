#!/usr/bin/env bash

bash benchmarks/dist_train_linear.sh configs/benchmarks/fyp/imagenet_linear_classification/r50_last.py ./work_dirs/extracted_weights/odc_r50_v1_epoch_400.pth
bash benchmarks/dist_train_linear.sh configs/benchmarks/fyp/imagenet_linear_classification/r50_last.py ./work_dirs/extracted_weights/simclr_r50_bs128_ep200_epoch200.pth
bash benchmarks/dist_train_linear.sh configs/benchmarks/fyp/imagenet_linear_classification/r50_last.py ./work_dirs/extracted_weights/simclr_r50_bs128_ep200_lr2_gpu1_epoch_200.pth
bash benchmarks/dist_train_linear.sh configs/benchmarks/fyp/imagenet_linear_classification/r50_last.py ./work_dirs/extracted_weights/contrastive_odc_r50_v7_2_bs128_epoch_200.pth
bash benchmarks/dist_train_linear.sh configs/benchmarks/fyp/imagenet_linear_classification/r50_last.py ./work_dirs/extracted_weights/contrastive_odc_r50_v7_2_bs128_lr2_epoch_200.pth



