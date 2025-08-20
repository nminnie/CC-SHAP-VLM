#!/bin/bash
#
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00
#SBATCH --account=tc067-s2737499

source /work/tc067/tc067/s2737499/vlm-env/bin/activate

srun ./translate.py results/xgqa_llava_onevision_pt_compare_20.json pt