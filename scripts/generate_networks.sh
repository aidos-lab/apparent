#!/bin/bash

#SBATCH -o ./log_files/network-gen_output.txt
#SBATCH -e ./log_files/network-gen_error.txt
#SBATCH -J Wayland_doc-orc_network-gen
#SBATCH -p cpu_p
#SBATCH -c 1
#SBATCH --mem=30G
#SBATCH -t 01:00:00
#SBATCH --nice=10000 

# You can put arbitrary unix commands here, call other scripts, etc...

poetry run python ../doc_orc/build_networks.py
echo "Finished Generating Networks!"


