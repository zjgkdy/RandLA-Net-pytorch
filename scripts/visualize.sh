#!/bin/bash
export PYTHONPATH=$(pwd) 

Dataset=results/checkpoint_0723/random_sampler_predictions
Sequence=08
python ./tools/visualize_SemanticKITTI.py \
    --dataset $Dataset \
    --sequence $Sequence 