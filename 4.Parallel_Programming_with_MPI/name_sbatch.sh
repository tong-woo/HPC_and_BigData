#!/bin/bash -e
#SBATCH -t 5:30
#SBATCH -N 16 --ntasks-per-node=1

mpirun ./mat_vec_mul
