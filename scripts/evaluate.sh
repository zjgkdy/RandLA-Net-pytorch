#!/bin/bash
export PYTHONPATH=$(pwd) 

EvalType=sub
Dataset=results/checkpoint_0727/random_sampler_movable 
Sequence=08
Datacfg=config/semantic-kitti-movable.yaml

python ./tools/evaluate_SemanticKITTI.py \
    --eval_type $EvalType\
    --dataset $Dataset \
    --sequence $Sequence \
    --datacfg $Datacfg