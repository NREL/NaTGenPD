#!/bin/bash

cd /home/mrossol/NaTGenPD/bin

qstat -u mrossol | grep CEMS | grep ' [RQ] ' || qsub -N CEMS run_CEMS.sh
