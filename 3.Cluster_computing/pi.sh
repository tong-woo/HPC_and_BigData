#!/bin/bash
#SBATCH --job-name="pi"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --time=00:10:00
#SBATCH --partition=normal
#SBATCH --output=pi_%j.out
#SBATCH --error=pi_%j.err

module purge
module load 2020
module load GCC/9.3.0

echo "OpenMP parallelism"
echo

for ncores in {4..24}

do
  export OMP_NUM_THREADS=$ncores

  echo "CPUS: " $OMP_NUM_THREADS
  echo "CPUS: " $OMP_NUM_THREADS >&2
  ./pi
  echo "DONE "
done
