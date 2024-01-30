#!/bin/bash

for i in 1 2 4 8 16 32
do 
    echo "[configuration]: OMP_NUM_THREADS=$i $1"
    OMP_NUM_THREADS=$i srun --nodes=1 $1
done

