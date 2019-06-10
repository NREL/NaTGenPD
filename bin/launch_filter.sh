#!/bin/bash

cd /home/mrossol/NaTGenPD/bin

qstat -u mrossol | grep CEMS | grep ' [RQ] ' || qsub -N CEMS filter_CEMS.sh
