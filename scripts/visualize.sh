#!/bin/bash
export PYTHONPATH=$(pwd) 

Dataset=log/debug/predictions
Sequence=08
python ./tools/visualize_SemanticKITTI.py \
    --dataset $Dataset \
    --sequence $Sequence 