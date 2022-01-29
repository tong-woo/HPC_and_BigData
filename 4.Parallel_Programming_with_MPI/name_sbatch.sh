#!/bin/bash -e
#SBATCH -t 3:00 --mem=100M 
#SBATCH -N 1 --ntasks-per-node=1
#SBATCH --reservation=course-jhlsrf011-0


mpirun ./mat_vec_mul
