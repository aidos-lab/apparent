#!/bin/bash -x
#SBATCH --account=hai_ricci
# budget account where contingent is taken from
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=10
# if keyword omitted: Max. 96 tasks per node
# (SMT enabled, see comment below)
#SBATCH --cpus-per-task=1
# for OpenMP/hybrid jobs only
#SBATCH --output=/p/project/hai_ricci/wayland1/aidos/doc-orc/scripts/log_files/slurm-%j.out
# if keyword omitted: Default is slurm-%j.out in
# the submission directory (%j is replaced by
# the job ID).
#SBATCH --error=/p/project/hai_ricci/wayland1/aidos/doc-orc/scripts/log_files/slurm-%j.error
# if keyword omitted: Default is slurm-%j.out in
# the submission directory.
#SBATCH --time=00:01:00
#SBATCH --partition=booster
#SBATCH --gres=gpu:0

conda activate doc-orc

python ../doc_orc/build_networks.py
echo "Finished Generating Networks!"


