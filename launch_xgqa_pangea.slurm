#!/bin/bash
#
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:2
#SBATCH --time=04:00:00
#SBATCH --account=tc067-s2737499

source /work/tc067/tc067/s2737499/lv-env/bin/activate

srun --ntasks=1 ./run-xgqa.py xgqa pangea 1000 1 data/ ru compare
