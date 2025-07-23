#!/bin/bash
Dataset=log/debug/predictions
Sequence=08
python visualize_SemanticKITTI.py \
    --dataset $Dataset \
    --sequence $Sequence 