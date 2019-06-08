#!/bin/bash
#PBS -l nodes=1:ppn=24,walltime=48:00:00,qos=high
#PBS -A naris
#PBS -q batch-h
#PBS -e $PBS_JOBNAME-$PBS_JOBID.err
#PBS -o $PBS_JOBNAME-$PBS_JOBID.out

python run_CEMS.py
