#!/bin/bash
#SBATCH --ntasks 4
#SBATCH --job-name MalwareDiffusion
#SBATCH --partition=k2-gpu
#SBATCH --gres gpu:1
#SBATCH --mem-per-cpu=16G

module add apps/python3/3.10.5/gcc-9.3.0
module add libs/nvidia-cuda/11.7.0/bin

export PYTHONPATH=$PYTHONPATH:/users/40237845/gridware/share/python/3.10.5/lib/python3.10/site-packages/

srun --ntasks=1 python3 main.py 0.001 100 100 &
srun --ntasks=1 python3 main.py 0.001 100 50 &
srun --ntasks=1 python3 main.py 0.001 100 25 &
srun --ntasks=1 python3 main.py 0.001 100 1 &
wait