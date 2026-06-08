#!/bin/bash

# Parse command line arguments
CONFIG_ID=$1
SWEEP_CONFIG_FILE=$2

# Load the specific configuration
python3 -c "
import sys
sys.path.append('.')
from sweep_configs import SWEEP_CONFIGS
config = SWEEP_CONFIGS[$CONFIG_ID]

# Build command line arguments
args = []
for key, value in config.items():
    if key == 'experiment_name':
        args.append(f'--experiment_name={value}')
    elif isinstance(value, bool):
        if value:
            args.append(f'--{key}')
    else:
        args.append(f'--{key}={value}')

# Write to temporary file
with open(f'temp_args_{$CONFIG_ID}.txt', 'w') as f:
    f.write(' '.join(args))
"
NUM_ENV_STEPS=4000000
NUM_AGENTS=1
PERCEPTION_TYPE="projected"
EVAL_ROLLOUT_THREADS=10
EVAL_EP_LENGTH=800
EVAL_RENDER_EPISODES=20
EXP_FOLDER_NAME=check
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Read the arguments and run training
ARGS=$(cat temp_args_${CONFIG_ID}.txt)
RUN_NAME=$(echo "$ARGS" | grep -o -- '--run_name=[^[:space:]]*' | cut -d'=' -f2 | tr -d '"')
RUN_FOLDER=${TIMESTAMP}_${NUM_AGENTS}_${RUN_NAME}
USER=$(whoami)
case "$USER" in
    ramalik)
        USER_RESULTS_DIR="ramalik/"
        ;;
    sjohnsonyu)
        USER_RESULTS_DIR="sonja_results/"
        ;;
    nwu)
        USER_RESULTS_DIR="nwu/"
        ;;
    satsingh)
        USER_RESULTS_DIR="satsingh/"
        ;;
    *)
        USER_RESULTS_DIR=""
        ;;
esac


RESULTS_PARENT_DIR="/n/holylfs06/LABS/krajan_lab/Lab/zfish/${USER_RESULTS_DIR}"
# RESULTS_PARENT_DIR="./"
RUN_DIR=${RESULTS_PARENT_DIR}results/rmappo-MultiAgentForagingEnv-${EXP_FOLDER_NAME}/${RUN_FOLDER}

echo "Running training with config $CONFIG_ID: $ARGS"

python train_ZFish.py --num_env_steps=$NUM_ENV_STEPS \
                            --num_agents=$NUM_AGENTS \
                            --timestamp=$TIMESTAMP \
                            --use_1dof_eyes \
                            --results_parent_dir $RESULTS_PARENT_DIR \
                      $ARGS

python eval_ZFish.py $RUN_DIR \
                    --n_rollout_threads $EVAL_ROLLOUT_THREADS \
                    --eval_episode_length $EVAL_EP_LENGTH \
                    --eval_render_episodes $EVAL_RENDER_EPISODES \
                    --save_vids \
                    --num_vids_to_save=1 \
                    --eval_eating_distribution_decay=5 \
                    --eval_uniform_reset_food_density=0.003 \
                    --eval_uniform_max_food_density=0.004 \
                    --eval_arena_size_min=100,100 \
                    --eval_arena_size_max=100,100 \
                    --eval_num_walkerbots=0

python preprocess_flatten.py $RUN_DIR

python report_behavior_shs.py $RUN_DIR/outputs

# python hunting_traces_nw.py $RUN_DIR/outputs

# python bout_analysis_nw.py $RUN_DIR/outputs

# python report_movement.py $RUN_DIR/outputs

# python report_vergence.py $RUN_DIR/outputs

# python report_rnn.py $RUN_DIR/outputs

# bash run_additional_exps.sh $RUN_FOLDER

# python gen_virtual_exps_figures.py $RUN_DIR/outputs # Optional: Regenerate VE figures from CSV

# Cleanup
rm temp_args_${CONFIG_ID}.txt

echo "Completed training and evaluating for config $CONFIG_ID"