#!/bin/bash
#SBATCH --job-name=test_py
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:05:00
#SBATCH --output=test_%j.out

module load Python/3.11.5     # ← 改成存在的版本
which python3 || echo "python3 not found"
python3 --version || echo "python3 failed"
python --version  || echo "python failed"
