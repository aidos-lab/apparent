#!/bin/bash

#SBATCH -o ./log_files/installer_output.txt
#SBATCH -e ./log_files/isntaller_error.txt
#SBATCH -J WaylandInstallation
#SBATCH -p cpu_p
#SBATCH -c 1
#SBATCH --mem=6G
#SBATCH -t 00:20:00
#SBATCH --nice=10000 

# You can put arbitrary unix commands here, call other scripts, etc...

poetry install
echo "Poetry works!"


