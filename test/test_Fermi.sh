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
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1
#SBATCH --mem=2G



#SBATCH --job-name="Test"
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

cd /home/PERSONALE/jinghao.wang2/code/test


python test_Fermi.py
# python test_mu.py
# python test_interaction.py

echo "Job finished at $(date)"
