#!/bin/bash

# 定义变量
base_size=256
crop_size=256
st_model="DNANet"
# TODO: 这里的训练还没做
model_dir="NUDT-SIRST_DNANet_12_01_2024_23_05_14_wDS/mIoU__DNANet_NUDT-SIRST_epoch.pth.tar"
dataset="NUDT-SIRST"
split_method="50_50"
model="DNANet"
backbone="resnet_18"
deep_supervision="True"
test_batch_size=1
mode="TXT"

# 执行Python程序
python test.py \
    --base_size $base_size \
    --crop_size $crop_size \
    --st_model $st_model \
    --model_dir $model_dir \
    --dataset $dataset \
    --split_method $split_method \
    --model $model \
    --backbone $backbone \
    --deep_supervision $deep_supervision \
    --test_batch_size $test_batch_size \
    --mode $mode
