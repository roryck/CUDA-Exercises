#!/bin/bash
# Example submit command: qsub -V -l h_rt=00:05:00 -P gpgpu -q normal -pe 1way 8 run-longhorn.sh

cd $HOME/CUDA-Exercises/dot_product

./dot_prod > dot_prod.out
