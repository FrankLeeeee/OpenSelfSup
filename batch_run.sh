#!/usr/bin/env bash

bash benchmarks/dist_train_linear.sh configs/benchmarks/fyp/imagenet_linear_classification/r50_last.py ./work_dirs/extracted_weights/contrastive_odc_r50_v17_bs128_epoch_200.pth
bash benchmarks/dist_train_linear.sh configs/benchmarks/fyp/imagenet_linear_classification/r50_last.py ./work_dirs/extracted_weights/contrastive_odc_r50_v18_bs128_epoch_200.pth



