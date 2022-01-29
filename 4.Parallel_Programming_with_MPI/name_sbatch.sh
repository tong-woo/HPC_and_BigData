#!/bin/bash -e
#SBATCH -t 5:00 
#SBATCH -N 2 --ntasks-per-node=1

mpirun ./mat_vec_mul
