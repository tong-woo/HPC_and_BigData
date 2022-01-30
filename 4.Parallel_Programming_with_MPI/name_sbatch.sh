#!/bin/bash -e
#SBATCH -t 1:10 
#SBATCH -N 2 --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=normal

export OMP_NUM_THREADS=16

mpirun ./mat_vec_mul
