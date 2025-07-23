#!/bin/bash
EvalType=sub
Dataset=log/debug/predictions
Sequence=08
Datacfg=config/semantic-kitti.yaml
python evaluate_SemanticKITTI.py \
    --eval_type $EvalType\
    --dataset $Dataset \
    --sequence $Sequence \
    --datacfg $Datacfg