#!/usr/bin/env bash

# for EPOCH in 40 80 120 160 200
# do
#     python tools/extract_backbone_weights.py ./work_dirs/fyp/100_cls_dataset/odc_r50_v1_lr002_classes500_gpu1/epoch_${EPOCH}.pth  ./work_dirs/extracted_weights/odc_r50_v1_lr002_classes500_gpu1_epoch_${EPOCH}.pth
#     bash benchmarks/dist_train_linear.sh configs/benchmarks/fyp/imagenet_linear_classification/r50_last.py ./work_dirs/extracted_weights/odc_r50_v1_lr002_classes500_gpu1_epoch_${EPOCH}.pth
# done


for SIZE in 32 64 256
do
    python tools/extract_backbone_weights.py ./work_dirs/fyp/100_cls_dataset/contrastive_odc_r50_v24_bs128_lr2_classes100_mincluster${SIZE}/epoch_200.pth  ./work_dirs/extracted_weights/contrastive_odc_r50_v24_bs128_lr2_classes100_mincluster${SIZE}_epoch_200.pth
    bash benchmarks/dist_train_linear.sh configs/benchmarks/fyp/imagenet_linear_classification/r50_last.py ./work_dirs/extracted_weights/contrastive_odc_r50_v24_bs128_lr2_classes100_mincluster${SIZE}_epoch_200.pth
done