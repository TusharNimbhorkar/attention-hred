#!/bin/bash
#SBATCH -n 1
#SBATCH -p gpu
#SBATCH -t 13:00:00
#SBATCH -o fib_%A.output
#SBATCH -e fib_%A.error

module load python/3.5.2
module load cuda/8.0.44
module load cudnn/8.0-v6.0

pip3 install --user --upgrade numpy
python3 train_model.py
