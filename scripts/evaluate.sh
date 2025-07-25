#!/bin/bash
export PYTHONPATH=$(pwd) 

EvalType=sub
Dataset=results/checkpoint_0722/crop_sampler_predictions  
Sequence=08
Datacfg=config/semantic-kitti.yaml

python ./tools/evaluate_SemanticKITTI.py \
    --eval_type $EvalType\
    --dataset $Dataset \
    --sequence $Sequence \
    --datacfg $Datacfg