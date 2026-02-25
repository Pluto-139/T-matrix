#!/bin/bash
#---------------------------------------------------------------------------- #
#   University    |   DIFA - Dept of Physics and Astrophysics 
#       of        |   Open Physics Hub
#    Bologna      |   (https://site.unibo.it/openphysicshub/en)
#---------------------------------------------------------------------------- #
# Author
#   Carlo Cintolesi (Template) & Gemini (Adaptation)
# --------------------------------------------------------------------------- #
# SLURM setup
# --------------------------------------------------------------------------- #

#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=112
#SBATCH --ntasks-per-node=1
#SBATCH --mem=256G



#SBATCH --job-name="Continuous_T_matrix"
#SBATCH --output=%N_%j.out
#SBATCH --error=%N_%j.err
#SBATCH --mail-type=ALL

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# --------------------------------------------------------------------------- #
# Modules setup and applications run
# --------------------------------------------------------------------------- #

module purge
module load astro/python/3.10.0     


echo "Job started at $(date)"
echo "Running on host: $(hostname)"
echo "Allocated CPUs: $SLURM_CPUS_PER_TASK"

cd /home/PERSONALE/jinghao.wang2/code


python t_con_hubbard.py

echo "Job finished at $(date)"
