#!/bin/bash

# Replace /path/to/conda with the actual path to your conda installation
source /hpc2hdd/home/mgong081/anaconda3/etc/profile.d/conda.sh

# Activate your conda environment
conda activate PytorchDeepSynergy
python src/model/run.py
