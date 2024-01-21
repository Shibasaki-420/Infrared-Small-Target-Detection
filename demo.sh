#!/bin/bash

# 定义变量
base_size=256
crop_size=256
img_demo_dir='img_demo'
img_demo_index='target1'
model='DNANet'
backbone='resnet_18'
deep_supervision='True'
test_batch_size=1
mode='TXT'
suffix='.png'

# 执行Python程序
python demo.py \
    --base_size $base_size \
    --crop_size $crop_size \
    --img_demo_dir $img_demo_dir \
    --img_demo_index $img_demo_index \
    --model $model \
    --backbone $backbone \
    --deep_supervision $deep_supervision \
    --test_batch_size $test_batch_size \
    --mode $mode \
    --suffix $suffix

