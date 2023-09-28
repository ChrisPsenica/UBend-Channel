#!/bin/bash

# Copy/paste this job script into a text file and submit with the command: 
# sbatch filename

#SBATCH --time=120:00:00   # walltime limit (HH:MM:SS)
#SBATCH --nodes=2   # number of nodes
#SBATCH --ntasks-per-node=36   # 36 processor core(s) per node 
#SBATCH --job-name="opt"
#SBATCH --output="log-%j.txt" # job standard output file (%j replaced by job id)

# LOAD MODULES
. /work/phe/DAFoam_Nova_Gcc/latest/loadDAFoam.sh


# RUN CODES
sh preProcessing.sh
mpirun -np 72 python runScript.py -optimizer=SNOPT
