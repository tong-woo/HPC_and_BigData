#!/bin/bash -e
#SBATCH -t 1:10 --mem=94000
#SBATCH -N 2 --ntasks-per-node=1

mpirun ./mat_vec_mul
