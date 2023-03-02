#!/bin/bash

#SBATCH -o poetry_output.txt
#SBATCH -e poetry_error.txt
#SBATCH -J WaylandInstallation
#SBATCH -p cpu_p
#SBATCH -c 1
#SBATCH --mem=6G
#SBATCH -t 00:20:00
#SBATCH --nice=10000 

# You can put arbitrary unix commands here, call other scripts, etc...

cd ../Dev/doc-orc/
poetry shell
export LIBGIT2=$VIRTUAL_ENV
cd /home/aih/jeremy.wayland/.local/bin/libgit2
cmake . -DCMAKE_INSTALL_PREFIX=$LIBGIT2
cmake --build . --target install


