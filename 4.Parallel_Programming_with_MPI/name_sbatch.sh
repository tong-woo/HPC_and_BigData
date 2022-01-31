#!/bin/bash -e
#SBATCH -t 5:00:00
#SBATCH --nodes=1 
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=normal


for thread in {1..16}
do
    export OMP_NUM_THREADS=$thread
    ./mat_vec_mul
done



#export OMP_NUM_THREADS=16

#./mat_vec_mul
