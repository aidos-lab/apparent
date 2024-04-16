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



DATA=../data/test_sample.pkl
alpha=0

poetry run python ../src/analysis.py --data ${DATA} --alpha ${alpha}
echo "Finished Generating Networks!"