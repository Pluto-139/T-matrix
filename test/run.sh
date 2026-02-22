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

#SBATCH --qos=debug          
#SBATCH --ntasks=1           
#SBATCH --nodes=1            
#SBATCH --cpus-per-task=8   
#SBATCH --mem-per-cpu=2G     

#SBATCH --job-name="Hubbard_MC" 
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


which python
which python3
python --version
python3 --version

echo "Job started at $(date)"
cd /home/PERSONALE/jinghao.wang2/Code

python hubbard_cluster.py


echo "Job finished at $(date)"
