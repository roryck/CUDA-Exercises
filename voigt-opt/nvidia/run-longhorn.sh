#!/bin/bash
# Example submit command: qsub -V -l h_rt=00:05:00 -P gpgpu -q development -pe 1way 8 run-longhorn.sh

cd $HOME/nvidia

./test_voigt > voigt.out
