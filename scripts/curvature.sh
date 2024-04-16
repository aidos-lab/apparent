#!/bin/bash -x
#SBATCH --account=hai_ricci
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=10
#SBATCH --cpus-per-task=1
#SBATCH --output=/p/project/hai_ricci/wayland1/aidos/doc-orc/scripts/log_files/curvature-%j.out
#SBATCH --error=/p/project/hai_ricci/wayland1/aidos/doc-orc/scripts/log_files/curvature-%j.error
#SBATCH --time=02:00:00
#SBATCH --partition=booster
#SBATCH --gres=gpu:0



DATA=../data/choc_curvature_features.pkl
alpha=0

poetry run python ../apparent/compute_features.py --data ${DATA} --alpha ${alpha} -c Forman -s --sample_size=1000
echo "Finished Generating Networks!"