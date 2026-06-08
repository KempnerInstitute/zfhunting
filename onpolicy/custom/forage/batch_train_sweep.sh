#!/bin/bash
#SBATCH -c 16
#SBATCH -p kempner
#SBATCH -t 1-12:00:00 
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus-per-node=1
#SBATCH --account=kempner_krajan_lab
#SBATCH --constraint=cc7.0|cc8.0|cc8.6
#SBATCH --exclude=holygpu8a[19604,19605,19606]
#SBATCH -J ma_zfish_clustertest_0
#SBATCH -o ./logs/MAZFish_cluster_test_0/ma_zfish_clustertest_0_%a.out
#SBATCH -e ./logs/MAZFish_cluster_test_0/ma_zfish_clustertest_0_%a.err
#SBATCH --array=0-19  # Adjust based on number of configs in SWEEP_CONFIGS

mkdir -p ./logs/MAZFish_cluster_test_0

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Running on node: $HOSTNAME"
echo "Started at: $(date)"

module load cuda/11.8.0-fasrc01
source activate zfish

export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

bash run_single_train.sh $SLURM_ARRAY_TASK_ID sweep_configs.py

echo "Finished at: $(date)"
