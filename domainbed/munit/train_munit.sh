##!/bin/bash
#
## Path to MUNIT configuration file.  Edit this file to change the number of iterations,
## how frequently checkpoints are saved, and other properties of MUNIT.
## The parameter `style_dim` corresponds to the dimension of `delta` in our work.
#export CONFIG_PATH=./core/tiny_munit.yaml
#
## Output images and checkpoints will be saved to this path.
#export OUTPUT_PATH=./results-mnist
#
#export CUDA_VISIBLE_DEVICES=0
#python3 train_munit.py --config $CONFIG_PATH --output_path $OUTPUT_PATH


export PYTHONPATH=$PYTHONPATH:"/home/hxw171930/mbdg-clean"
output_path_prefix="saved_models/VLCSOOD/test_env"
ood_class=0
gpu_start_index=0

#declare -a test_env_settings=("0" "1" "2" "3")
#declare -a test_env_settings=("3")
#for test_envs in "${test_env_settings[@]}"
#do
#  nohup python train_munit.py --dataset VLCS --output_path "$output_path_prefix"_"$test_envs" --test_envs "$(("$test_envs"))" --ood_classes "$ood_class" --device $((gpu_start_index % 4)) --resume > saved_models/VLCSOOD/out_"$test_envs".txt &
#  ((gpu_start_index++))
#done

#declare -a test_env_settings=("01" "02" "03" "12" "13" "23")
declare -a test_env_settings=("01" "23")
for test_envs in "${test_env_settings[@]}"
do
  test_env1=$(( ${test_envs:0:1} ))
  test_env2=$(( ${test_envs:1:1} ))
  nohup python train_munit.py --dataset VLCS --output_path "$output_path_prefix"_"$test_envs" --test_envs "$test_env1" "$test_env2" --ood_classes "$ood_class" --device $((gpu_start_index % 4)) --resume > saved_models/VLCSOOD/out_"$test_envs".txt &
  ((gpu_start_index++))
done
