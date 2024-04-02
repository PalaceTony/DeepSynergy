#!/bin/bash

source /hpc2hdd/home/mgong081/anaconda3/etc/profile.d/conda.sh
conda activate PytorchDeepSynergy

# Lists to run
datasets=("data_test_fold0_tanh.p" "data_test_fold1_tanh.p" "data_test_fold2_tanh.p" "data_test_fold3_tanh.p" "data_test_fold4_tanh.p")
models=("deepSynergy" "3MLP" "matchMaker")

for model in "${models[@]}"; do
    for dataset in "${datasets[@]}"; do
        dataset="/hpc2hdd/home/mgong081/Projects/DeepSynergy/data/$dataset"
        python src/model/run.py --epochs 2 --model $model --data_file=$dataset
    done
done
