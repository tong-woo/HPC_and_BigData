#!/bin/bash -e
#SBATCH -t 10:00
#SBATCH --nodes=1 
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=normal

export OMP_NUM_THREADS=16

./mat_vec_mul
