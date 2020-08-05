#!/bin/bash -l
#SBATCH -N 1 --ntasks-per-node=20
#SBATCH --mail-user=grabka@ucsb.edu
#SBATCH --mail-type=start,end
cd seagul
git pull origin rbf
cd ..
module load singularity
singularity exec $HOME/rbf.simg python3.7 $1
