#!/bin/bash

# Replace /path/to/conda with the actual path to your conda installation
source /hpc2hdd/home/mgong081/anaconda3/etc/profile.d/conda.sh

# Activate your conda environment
conda activate PytorchDeepSynergy

# List of dataset files to iterate over
datasets=("data_test_fold0_tanh.p" "data_test_fold1_tanh.p" "data_test_fold2_tanh.p" "data_test_fold3_tanh.p" "data_test_fold4_tanh.p")

# Base path to the datasets
base_path="/hpc2hdd/home/mgong081/Projects/DeepSynergy/data"

for dataset in "${datasets[@]}"; do
    # Construct the full path to the dataset
    data_path="$base_path/$dataset"
    
    # Run the Python script with the current dataset
    python src/model/run.py data.file="$data_path"
done
