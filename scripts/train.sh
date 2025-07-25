#!/bin/bash
export PYTHONPATH=$(pwd) 

ModelCfg=random_sampler_config
Dataset=data/dataset/sequences_0.06
CheckpointPath=pretrain_model/checkpoint.tar
LogDir=./log/random_sample_0723
MaxEpoch=100
TrainBatchSize=4
ValBatchSize=20
NumWorkers=10

export DEBUG=1
export CUDA_VISIBLE_DEVICES=0

python ./tools/train_SemanticKITTI.py \
    --model_config $ModelCfg \
    --dataset $Dataset \
    --log_dir $LogDir \
    --max_epoch $MaxEpoch \
    --batch_size $TrainBatchSize \
    --val_batch_size $ValBatchSize \
    --num_workers $NumWorkers \
   # --checkpoint_path $CheckpointPath \
