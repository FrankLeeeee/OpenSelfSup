#!/usr/bin/env bash
python tools/extract_backbone_weights.py ./work_dirs/fyp/100_cls_dataset/contrastive_odc_r50_v21_bs128/epoch_200.pth ./work_dirs/extracted_weights/contrastive_odc_r50_v21_bs128_epoch_200.pth
python tools/extract_backbone_weights.py ./work_dirs/fyp/100_cls_dataset/contrastive_odc_r50_v22_bs128/epoch_200.pth ./work_dirs/extracted_weights/contrastive_odc_r50_v22_bs128_epoch_200.pth

bash benchmarks/dist_train_linear.sh configs/benchmarks/fyp/imagenet_linear_classification/r50_last.py ./work_dirs/extracted_weights/contrastive_odc_r50_v21_bs128_epoch_200.pth
bash benchmarks/dist_train_linear.sh configs/benchmarks/fyp/imagenet_linear_classification/r50_last.py ./work_dirs/extracted_weights/contrastive_odc_r50_v22_bs128_epoch_200.pth



