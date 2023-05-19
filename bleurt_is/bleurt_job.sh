#!/bin/sh
#SBATCH --job-name=SCORES_BLEURT_D12_file_1
#SBATCH -N 1
#SBATCH -n 14    ##14 cores(of28) so you get 1/2 of machine RAM (64 GB of 128GB)
#SBATCH --gres=gpu:0   ## Run on 1 GPU
#SBATCH --output job%j.out
#SBATCH --error job%j.err
#SBATCH --mail-user=anmol.agarwal@students.iiit.ac.in,shrey.gupta@students.iiit.ac.in
#SBATCH --mail-type=ALL
#SBATCH -p AI_Center,gpu-v100-16gb,gpu-v100-32gb


##Load your modules and run code here
hostname
module load cuda/11.1

export CUDA_VISIBLE_DEVICES=0

module load python3/anaconda/2020.02

source activate /work/shreyg/ENVS/bleurt_env

echo "performing unit test"

python -m unittest bleurt.score_test

echo "now starting to run"

python /home/shreyg/bleurt.py model_1_rev.json

conda deactivate