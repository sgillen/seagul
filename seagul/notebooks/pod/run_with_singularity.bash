#!/bin/bash -l
#SBATCH -N 1 --ntasks-per-node=40
#SBATCH --mail-user=sgillen@ucsb.edu
#SBATCH --mail-type=start,end

git pull
module load singularity
singularity exec $HOME/rllib.simg python3 $1
