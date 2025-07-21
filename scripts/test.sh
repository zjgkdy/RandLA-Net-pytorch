#!/bin/bash
export PYTHONPATH=$(pwd) 

ModelCfg=random_sampler_config
Dataset=data/dataset/sequences_0.06
CheckpointPath=pretrain_model/random_sampler_movable_checkpoint.tar
ResultDir=results/checkpoint_0727/random_sampler_movable 
YamlConfig=config/semantic-kitti-movable.yaml
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