#!/bin/bash
export PYTHONPATH=$(pwd) 

ModelCfg=crop_sampler_config
Dataset=data/dataset/sequences_0.06
CheckpointPath=pretrain_model/crop_sampler_checkpoint_0722.tar
ResultDir=results/checkpoint_0722/crop_sampler_predictions  
YamlConfig=config/semantic-kitti.yaml
BatchSize=20
NumWorkers=10

export CUDA_VISIBLE_DEVICES=0

python ./tools/test_SemanticKITTI.py \
    --model_config $ModelCfg \
    --dataset $Dataset \
    --checkpoint_path $CheckpointPath \
    --result_dir $ResultDir \
    --yaml_config $YamlConfig \
    --batch_size $BatchSize \