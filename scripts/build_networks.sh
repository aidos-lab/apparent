#!/bin/bash -x
#SBATCH --account=hai_ricci
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=96
#SBATCH --output=/p/project/hai_ricci/wayland1/aidos/doc-orc/scripts/log_files/build_networks-%j.out
#SBATCH --error=/p/project/hai_ricci/wayland1/aidos/doc-orc/scripts/log_files/build_networks-%j.error
#SBATCH --time=04:00:00
#SBATCH --partition=booster
#SBATCH --gres=gpu:0

source ~/.bashrc 
conda activate doc-orc

python ../doc_orc/build_networks.py
echo "Finished Generating Networks!"


