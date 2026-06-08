#!/usr/bin/env bash
set -euo pipefail

# ---- Fixed knobs you had in your snippet ----
SEED=1
uniform_reset_food_density=0.012
NUM_ENV_STEPS=300000
# NUM_ENV_STEPS=2000000
EVAL_EP_LENGTH=800
NUM_AGENTS=1
PERCEPTION_TYPE="projected"
EVAL_ROLLOUT_THREADS=12
EVAL_RENDER_EPISODES=12
EVAL_NUM_VIDS=3
EXP_FOLDER_NAME="check"

# These were fixed in your snippet; keep as-is unless you want to sweep them too
CURR="time_normalized_step"
BOTS_EVAL=2
VERGENCE_MULTIPLIER=0
R_EAT=100

# Helper to make float safe for filenames (e.g., 0.1 -> 0p1)
float_tag () { echo "$1" | sed 's/\./p/g'; }

for BOTS_TRAIN in 0 1 2 3 4 5; do
  for BIN_ANGLE in skip on; do
    for NOISE in 0 0.1 0.2; do
      for USE_1DOF in skip on; do

        # Optional flags
        BINOCULAR_FLAG=""
        [ "$BIN_ANGLE" = "on" ] && BINOCULAR_FLAG="--binocular_angle_only"

        EYE1_FLAG=""
        [ "$USE_1DOF" = "on" ] && EYE1_FLAG="--use_1dof_eyes"

        # Tag fragments for run naming
        TAGS=""
        [ -n "$BINOCULAR_FLAG" ] && TAGS="${TAGS}_BA"
        [ -n "$EYE1_FLAG" ] && TAGS="${TAGS}_1dof"
        TAGS="${TAGS}_n$(float_tag "$NOISE")"

        TIMESTAMP=$(date +%Y%m%d_%H%M%S)

        RUN_NAME="GRU_${NUM_ENV_STEPS}_${BOTS_TRAIN}bots${BOTS_EVAL}_food012_fullturnpenalized_seed${SEED}_${CURR}_vm${VERGENCE_MULTIPLIER}_collision5${TAGS}"
        # RUN_NAME="Vanilla_${NUM_ENV_STEPS}_${BOTS_TRAIN}bots${BOTS_EVAL}_food012_fullturnpenalized_seed${SEED}_${CURR}_vm${VERGENCE_MULTIPLIER}_collision5${TAGS}"
        RUN_FOLDER="${TIMESTAMP}_${NUM_AGENTS}_${RUN_NAME}"
        RUN_DIR="results/rmappo-MultiAgentForagingEnv-${EXP_FOLDER_NAME}/${RUN_FOLDER}"
        mkdir -p "${RUN_DIR}"

        echo "=== Launching ${RUN_FOLDER} ==="

        python train_ZFish.py --num_env_steps="$NUM_ENV_STEPS" \
          --attn_mode "x+hx" --attn_use_softmax \
          --num_agents="$NUM_AGENTS" \
          --perception_type="$PERCEPTION_TYPE" \
          --n_rollout_threads 12 \
          --episode_length 400 --max_episode_length 400 \
          --binocular_depth_only \
          $BINOCULAR_FLAG \
          $EYE1_FLAG \
          --vergence_deviation="$(bc -l <<< "scale=5; -0.03 * $VERGENCE_MULTIPLIER")" \
          --run_name="$RUN_NAME" \
          --timestamp="$TIMESTAMP" \
          --food_detection_range=10 \
          --action_noise_std="$NOISE" \
          --seed="$SEED" \
          --uniform_reset_food_density="$uniform_reset_food_density" \
          --uniform_max_food_density=0.012 \
          --render_episodes 1 \
          --rnn_type gru \
          --curriculum_type "$CURR" \
          --curriculum_early_end_frac 0.9 \
          --r_eat_override "$R_EAT" \
          --walker_pursuit \
          --num_walkerbots="$BOTS_TRAIN"

        python eval_ZFish.py "$RUN_DIR" \
          --n_rollout_threads "$EVAL_ROLLOUT_THREADS" \
          --eval_episode_length "$EVAL_EP_LENGTH" \
          --eval_render_episodes "$EVAL_RENDER_EPISODES" \
          --save_vids \
          --num_vids_to_save="$EVAL_NUM_VIDS" \
          --eval_eating_distribution_decay=1 \
          --eval_uniform_reset_food_density=0.003 \
          --eval_uniform_max_food_density=0.004 \
          --eval_num_walkerbots="$BOTS_EVAL"

        python preprocess_flatten.py "$RUN_DIR"
        python report_behavior_shs.py "$RUN_DIR/outputs"
        python hunting_traces_nw.py "$RUN_DIR/outputs"
        python bout_analysis_nw.py "$RUN_DIR/outputs"
        python report_movement.py "$RUN_DIR/outputs"
        python report_vergence.py "$RUN_DIR/outputs"
        python report_rnn.py "$RUN_DIR/outputs"
        bash run_additional_exps.sh "$RUN_FOLDER"
        python report_virtual_exps.py "$RUN_DIR/outputs"

      done
    done
  done
done
