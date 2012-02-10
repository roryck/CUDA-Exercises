#!/bin/bash
# Example submit command: qsub run-longhorn.sh
#$ -A 20120209NCAR
#$ -V
#$ -l h_rt=00:05:00
#$ -cwd
#$ -j y
#$ -P gpgpu
#$ -q normal
#$ -pe 1way 8

./vec_rev > vec_rev.out
