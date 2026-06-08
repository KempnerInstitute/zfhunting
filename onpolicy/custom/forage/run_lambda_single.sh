NUM_ENV_STEPS=2000000
NUM_AGENTS=1
PERCEPTION_TYPE="projected"
EVAL_ROLLOUT_THREADS=10
EVAL_EP_LENGTH=400
EVAL_RENDER_EPISODES=30
EXP_FOLDER_NAME=check
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_NAME="nathan_test_long"
RUN_FOLDER=${TIMESTAMP}_${NUM_AGENTS}_${RUN_NAME}
RESULTS_PARENT_DIR="./"
RUN_DIR=${RESULTS_PARENT_DIR}results/rmappo-MultiAgentForagingEnv-${EXP_FOLDER_NAME}/${RUN_FOLDER}

# python train_ZFish.py --num_env_steps=$NUM_ENV_STEPS \
#                     --attn_mode "x+hx" --attn_use_softmax \
#                     --num_agents=$NUM_AGENTS \
#                     --binocular_depth_only \
#                     --use_1dof_eyes \
#                     --vergence_deviation=-0.003 \
#                     --binocular_angle_only \
#                     --run_name=$RUN_NAME \
#                     --timestamp=$TIMESTAMP \
#                     --food_detection_range=10 \
#                     --action_noise_std=0. \
#                     --results_parent_dir $RESULTS_PARENT_DIR

RUN_DIR=/home/nathanwu/zfish/onpolicy/custom/forage/results/rmappo-MultiAgentForagingEnv-check/20251121_120554_1_nathan_test_long

python eval_ZFish.py $RUN_DIR \
                    --n_rollout_threads $EVAL_ROLLOUT_THREADS \
                    --eval_episode_length $EVAL_EP_LENGTH \
                    --eval_render_episodes $EVAL_RENDER_EPISODES \
                    --save_vids \
                    --num_vids_to_save=1 \
                    --eval_eating_distribution_decay=5 \
                    --eval_uniform_reset_food_density=0.003 \
                    --eval_uniform_max_food_density=0.004 \
                    --eval_arena_size_min=100 \
                    --eval_arena_size_max=100 \
                    # --eval_num_walkerbots=2

python preprocess_flatten.py $RUN_DIR

# python report_behavior_shs.py $RUN_DIR/outputs

# python hunting_traces_nw.py $RUN_DIR/outputs

# python bout_analysis_nw.py $RUN_DIR/outputs

# #python report_movement.py $RUN_DIR/outputs

# python report_movement_nw.py $RUN_DIR/outputs

# python report_vergence.py $RUN_DIR/outputs

# python report_rnn.py $RUN_DIR/outputs

# bash run_additional_exps.sh $RUN_FOLDER

# python gen_virtual_exps_figures.py $RUN_DIR/outputs # Optional: Regenerate VE figures from CSV