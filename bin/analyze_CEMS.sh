#!/bin/bash
#PBS -l nodes=1,walltime=48:00:00,qos=high
#PBS -A naris
#PBS -q bigmem
#PBS -e $PBS_JOBNAME-$PBS_JOBID.err
#PBS -o $PBS_JOBNAME-$PBS_JOBID.out

python analyze_CEMS.py ${analysis}
