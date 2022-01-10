#!/bin/bash
#SBATCH -q regular
#SBATCH --time=02:00:00
###SBATCH -q debug
###SBATCH --time=00:15:00
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --constraint=haswell
#SBATCH -A m3900

export OMP_PROC_BIND=true
export OMP_PLACES=threads
export OMP_NUM_THREADS=32

conda activate cloud_cover

srun python pull_cloudless_bands_threaded.py
