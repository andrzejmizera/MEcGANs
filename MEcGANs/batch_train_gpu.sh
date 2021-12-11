#!/bin/bash -l
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --time=2-00:00:00
#SBATCH -p gpu
#SBATCH --gpus=1

CUDA_VISIBLE_DEVICES=0 ~/anaconda3/bin/python train_pix2pix.py --config_path configs/config_nirrgb2rgbcloud.yml --results_dir results

