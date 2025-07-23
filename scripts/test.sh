#!/bin/bash
Dataset=data/dataset/sequences_0.06
CheckpointPath=pretrain_model/checkpoint.tar
ResultDir=log/debug/predictions  
YamlConfig=config/semantic-kitti.yaml
BatchSize=20
NumWorkers=10

export DEBUG=0
export CUDA_VISIBLE_DEVICES=2,3

python test_SemanticKITTI.py \  
    --dataset $Dataset
    --checkpoint_path $CheckpointPath \
    --result_dir $ResultDir \
    --yaml_config $YamlConfig \
    --batch_size $BatchSize \