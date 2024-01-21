#!/bin/bash

# 定义变量
base_size=256
crop_size=256
epochs=1500
dataset="NUDT-SIRST"
split_method="50_50"
model="DNANet"
backbone="resnet_18"
deep_supervision="True"
train_batch_size=12
test_batch_size=12
mode="TXT"

# 执行Python程序
python train.py \
    --base_size $base_size \
    --crop_size $crop_size \
    --epochs $epochs \
    --dataset $dataset \
    --split_method $split_method \
    --model $model \
    --backbone $backbone \
    --deep_supervision $deep_supervision \
    --train_batch_size $train_batch_size \
    --test_batch_size $test_batch_size \
    --mode $mode