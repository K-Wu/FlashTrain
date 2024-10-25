#!/bin/bash
#SBATCH --time=0:59:00
#SBATCH -p gpuA100x4
#SBATCH --account=bcdt-delta-gpu
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --ntasks-per-node=4
#SBATCH --nodes=4
#SBATCH --mem=0
#SBATCH --job-name=do_dist_training
#SBATCH --output=my_output.%j.out
#SBATCH --error=my_output.%j.err

# Load any necessary modules (e.g., Anaconda)
export PYTHONPATH=$PYTHONPATH:$MY_TOOLKIT_PATH/../:$MY_TOOLKIT_PATH/../third_party/Megatron-DeepSpeed

# Activate your Python environment (if needed)
eval "$(conda shell.bash hook)"
conda activate dev_flashtrain
which python

# Run your Python script
# MY_TOOLKIT_PATH is set in sbatch_and_tail.sh and is /benchmark

bash third_party/delta_scripts/scale_pretrain_bert_distributed_with_mp_baseline.sh