#!/bin/bash

cd /home/mrossol/NaTGenPD/bin

for analysis in process quartiles
do
    qstat -u mrossol | grep CEMS_${analysis} | grep ' [RQ] ' || qsub -N CEMS_${analysis} -v analysis="${analysis}" analyze_CEMS.sh
done
