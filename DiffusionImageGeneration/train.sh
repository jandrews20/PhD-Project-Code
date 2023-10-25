#!/bin/bash
#SBATCH --job-name=1Ddiffusiontraining
#SBATCH -p k2-gpu
#SBATCH --gres gpu:1
#SBATCH --mem 16000M
#SBATCH --output=DiffusionTest-out-%j.out

module add apps/python3/3.10.5/gcc-9.3.0
module add libs/nvidia-cuda/11.7.0/bin


export PYTHONPATH=$PYTHONPATH:/users/40237845/gridware/share/python/3.10.5/lib/python3.10/site-packages/

python3 main.py
