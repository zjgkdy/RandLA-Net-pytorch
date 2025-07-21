#!/bin/bash
export PYTHONPATH=$(pwd) 

Config=config/semantic-kitti-movable.yaml
Dataset=results/checkpoint_0727/random_sampler_movable 
Sequence=08
python ./tools/visualize_SemanticKITTI.py \
    --config $Config \
    --dataset $Dataset \
    --sequence $Sequence 