#!/bin/bash

#SBATCH -o ./log_files/curvature_output.txt
#SBATCH -e ./log_files/curvature_error.txt
#SBATCH -J Wayland_doc-orc_curvature
#SBATCH -p cpu_p
#SBATCH -c 1
#SBATCH --mem=30G
#SBATCH -t 01:00:00
#SBATCH --nice=10000 

# You can put arbitrary unix commands here, call other scripts, etc...

DATA=../data/test_sample.pkl
alpha=0

poetry run python ../src/analysis.py --data ${DATA} --alpha ${alpha}
echo "Finished Generating Networks!"