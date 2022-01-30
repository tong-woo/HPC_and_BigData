#!/bin/bash -e
#SBATCH -t 1:00 
#SBATCH -N 1 --ntasks-per-node=16

mpirun ./mat_vec_mul
