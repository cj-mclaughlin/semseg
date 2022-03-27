#!/bin/bash
#SBATCH -p whitehill
#SBATCH --nodes=2
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=2

source ~/venv/bin/activate

export PYTHONPATH=./

PYTHON=python

dataset=$1
exp_name=$2
exp_dir=exp/${dataset}/${exp_name}
model_dir=${exp_dir}/model
result_dir=${exp_dir}/result
config=config/${dataset}/${dataset}_${exp_name}.yaml
now=$(date +"%Y%m%d_%H%M%S")

mkdir -p ${model_dir} ${result_dir}
cp tool/my_train.sh tool/my_train.py tool/test.sh tool/my_test.py ${config} ${exp_dir}

export PYTHONPATH=./
$PYTHON -u ${exp_dir}/my_train.py \
  --config=${config} \
  2>&1 | tee ${model_dir}/train-$now.log

# $PYTHON -u ${exp_dir}/my_test.py \
#   --config=${config} \
#   2>&1 | tee ${result_dir}/test-$now.log
