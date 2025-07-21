#!/bin/bash
export PYTHONPATH=$(pwd) 

SrcPath=data/dataset/sequences
DstPath=data/dataset/sequences_0.06
GridSize=0.06
YamlCfg=config/semantic-kitti-movable.yaml

python tools/data_prepare_semantickitti.py \
    --src_path $SrcPath \
    --dst_path $DstPath \
    --grid_size $GridSize \
    --yaml_config  $YamlCfg \