# for SEED in $(seq 6 10); do   
for SEED in 1; do   
    uniform_reset_food_density=0.012
    # NUM_ENV_STEPS=2000000; EVAL_EP_LENGTH=400
    # NUM_ENV_STEPS=1000000; EVAL_EP_LENGTH=400
    NUM_ENV_STEPS=300000; EVAL_EP_LENGTH=400
    # NUM_ENV_STEPS=100000; EVAL_EP_LENGTH=400
    # NUM_ENV_STEPS=20000; EVAL_EP_LENGTH=100 # Quick testing
    NUM_AGENTS=1
    PERCEPTION_TYPE="projected"
    EVAL_ROLLOUT_THREADS=12
    EVAL_RENDER_EPISODES=12 # 30
    EVAL_NUM_VIDS=3
    EXP_FOLDER_NAME=check
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)    
    # BIN_ANGLE="skip"
    BIN_DEPTH="skip"

    # BINOCULAR_ANGLE_FLAG=""
    # [ "$BIN_ANGLE" = "only" ] && BINOCULAR_ANGLE_FLAG="--binocular_angle_only"
    BINOCULAR_DEPTH_FLAG=""
    [ "$BIN_DEPTH" = "only" ] && BINOCULAR_DEPTH_FLAG="--binocular_depth_only"


    # CURR="fixed_step_with_max"
    CURR="time_normalized_step"

    # BOTS_TRAIN=1; BOTS_EVAL=2; VERGENCE_MULTIPLIER=0; R_EAT=100
    # BOTS_TRAIN=2; BOTS_EVAL=2; VERGENCE_MULTIPLIER=0; R_EAT=100
    # BOTS_TRAIN=3; BOTS_EVAL=3; VERGENCE_MULTIPLIER=0; R_EAT=100
    BOTS_TRAIN=3; BOTS_EVAL=3; VERGENCE_MULTIPLIER=2; R_EAT=100 # 0.06
    # BOTS_TRAIN=5; BOTS_EVAL=3; VERGENCE_MULTIPLIER=0; R_EAT=100
    # BOTS_TRAIN=5; BOTS_EVAL=3; VERGENCE_MULTIPLIER=1; R_EAT=100

    # BOTS_TRAIN=2; BOTS_EVAL=2; VERGENCE_MULTIPLIER=0; R_EAT=25
    # BOTS_TRAIN=0; BOTS_EVAL=0; VERGENCE_MULTIPLIER=0; R_EAT=25
    # BOTS_TRAIN=0; BOTS_EVAL=0; VERGENCE_MULTIPLIER=1; R_EAT=25

    RUN_NAME="MonoFoodNoisyAngleNoDepth_BAngle${BIN_ANGLE}_${NUM_ENV_STEPS}_${BOTS_TRAIN}bots${BOTS_EVAL}_food012_fullturnpenalized_seed${SEED}_${CURR}_vm${VERGENCE_MULTIPLIER}_reat${R_EAT}_Bot05x"
    # RUN_NAME="BDepth${BIN_ANGLE}_BAngle${BIN_ANGLE}_${NUM_ENV_STEPS}_${BOTS_TRAIN}bots${BOTS_EVAL}_food012_fullturnpenalized_seed${SEED}_${CURR}_vm${VERGENCE_MULTIPLIER}_reat${R_EAT}"
    # RUN_NAME="BAngle${BIN_ANGLE}_${NUM_ENV_STEPS}_${BOTS_TRAIN}bots${BOTS_EVAL}_food012_fullturnpenalized_seed${SEED}_${CURR}_vm${VERGENCE_MULTIPLIER}_collision5"
    # RUN_NAME="MonoAngle_GRU_${NUM_ENV_STEPS}_${BOTS_TRAIN}bots${BOTS_EVAL}_food012_fullturnpenalized_seed${SEED}_${CURR}_vm${VERGENCE_MULTIPLIER}_collision5"

    RUN_FOLDER=${TIMESTAMP}_${NUM_AGENTS}_${RUN_NAME}
    RUN_DIR=results/rmappo-MultiAgentForagingEnv-${EXP_FOLDER_NAME}/${RUN_FOLDER}
    mkdir -p ${RUN_DIR}

    python train_ZFish.py --num_env_steps=$NUM_ENV_STEPS \
        --attn_mode "x+hx" --attn_use_softmax \
        --num_agents=$NUM_AGENTS \
        --perception_type=$PERCEPTION_TYPE \
        --n_rollout_threads 12 \
        --episode_length 400 --max_episode_length 400 \
        $BINOCULAR_DEPTH_FLAG \
        --use_1dof_eyes \
        --vergence_deviation=$(bc -l <<< "scale=5; -0.03 * $VERGENCE_MULTIPLIER") \
        --run_name=$RUN_NAME \
        --timestamp=$TIMESTAMP \
        --food_detection_range=10 \
        --action_noise_std=0. \
        --seed=$SEED \
        --uniform_reset_food_density=$uniform_reset_food_density \
        --uniform_max_food_density=0.012 \
        --render_episodes 1 \
        --curriculum_type $CURR \
        --curriculum_early_end_frac 0.9 \
        --r_eat_override $R_EAT \
        --walker_pursuit \
        --binocular_distance_only --walker_monocular_perception --angle_noise_std_food=0.21 \
        --rnn_type vanilla \
        --num_walkerbots=$BOTS_TRAIN 
        # > ${RUN_DIR}/train_log.txt 2>&1 &
        # $BINOCULAR_ANGLE_FLAG \
        # --use_input_nonlinearity --use_output_nonlinearity \ 

    python eval_ZFish.py $RUN_DIR \
        --n_rollout_threads $EVAL_ROLLOUT_THREADS \
        --eval_episode_length $EVAL_EP_LENGTH \
        --eval_render_episodes $EVAL_RENDER_EPISODES \
        --save_vids \
        --num_vids_to_save=$EVAL_NUM_VIDS \
        --eval_eating_distribution_decay=1 \
        --eval_uniform_reset_food_density=0.003 \
        --eval_uniform_max_food_density=0.004 \
        --eval_num_walkerbots=$BOTS_EVAL
        # --walker_pursuit \

    # RUN_DIR="results/rmappo-MultiAgentForagingEnv-check/20250918_170932_1_GRU_300000_0bots0_food012_fullturnpenalized_seed1_time_normalized_step_vm3"
    # python preprocess_flatten.py $RUN_DIR
    # python report_behavior_shs.py $RUN_DIR/outputs
    # python hunting_traces_nw.py $RUN_DIR/outputs
    # python bout_analysis_nw.py $RUN_DIR/outputs
    # python report_movement.py $RUN_DIR/outputs
    # python report_vergence.py $RUN_DIR/outputs
    # python report_rnn.py $RUN_DIR/outputs
    # bash run_additional_exps.sh $RUN_FOLDER
    # python report_virtual_exps.py $RUN_DIR/outputs
done