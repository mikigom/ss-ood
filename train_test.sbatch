#!/bin/bash

source ~/new_begin.sh

echo "${IN}"
python train_OC_ImageNet.py -in ${IN}
python test_OC_ImageNet.py -in ${IN}
