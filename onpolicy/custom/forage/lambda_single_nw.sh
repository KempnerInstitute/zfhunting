# for SEED in $(seq 1); do   
for SEED in 2; do   
    NUM_ENV_STEPS=1000000
    NUM_AGENTS=4
    PERCEPTION_TYPE="projected"
    EVAL_ROLLOUT_THREADS=10
    EVAL_EP_LENGTH=400
    EVAL_RENDER_EPISODES=30
    EXP_FOLDER_NAME=check
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    RUN_NAME="collision1_nobots_seed${SEED}"
    RUN_FOLDER=${TIMESTAMP}_${NUM_AGENTS}_${RUN_NAME}
    RUN_DIR=results/rmappo-MultiAgentForagingEnv-${EXP_FOLDER_NAME}/${RUN_FOLDER}

    # python train_ZFish.py --num_env_steps=$NUM_ENV_STEPS \
    #                     --attn_mode "x+hx" --attn_use_softmax \
    #                     --num_agents=$NUM_AGENTS \
    #                     --perception_type=$PERCEPTION_TYPE \
    #                     --binocular_depth_only \
    #                     --use_1dof_eyes \
    #                     --vergence_deviation=-0.003 \
    #                     --binocular_angle_only \
    #                     --run_name=$RUN_NAME \
    #                     --timestamp=$TIMESTAMP \
    #                     --food_detection_range=10 \
    #                     --action_noise_std=0. \
    #                     --seed=$SEED \
    #                     --num_walkerbots=4

    # RUN_DIR=results/rmappo-MultiAgentForagingEnv-check/20250916_130702_1_collision1_nowalkerbots_seed1/
    # RUN_DIR=results/rmappo-MultiAgentForagingEnv-check/20250916_130731_1_collision1_nowalkerbots_seed2

    RUN_DIR=results/rmappo-MultiAgentForagingEnv-check/20250929_141009_4_collision1_nobots_seed2

    python eval_ZFish.py $RUN_DIR \
                        --n_rollout_threads $EVAL_ROLLOUT_THREADS \
                        --eval_episode_length $EVAL_EP_LENGTH \
                        --eval_render_episodes $EVAL_RENDER_EPISODES \
                        --save_vids \
                        --num_vids_to_save=1 \
                        --eval_eating_distribution_decay=1 \
                        --eval_uniform_reset_food_density=0.003 \
                        --eval_uniform_max_food_density=0.004 \
                        --eval_num_walkerbots=4

    # python preprocess_flatten.py $RUN_DIR

    python report_behavior_shs.py $RUN_DIR/outputs

    python hunting_traces_nw.py $RUN_DIR/outputs

    python bout_analysis_nw.py $RUN_DIR/outputs

    python report_movement_nw.py $RUN_DIR/outputs

    python report_vergence_nw.py $RUN_DIR/outputs

done
 