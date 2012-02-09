#!/bin/bash
# Example submit command: qsub -A 20120209NCAR -V -l h_rt=00:05:00 -P gpgpu -q development -pe 1way 8 run-longhorn.sh

cd $HOME/CUDA-Exercises/voigt-opt/nvidia

./test_voigt > voigt.out
