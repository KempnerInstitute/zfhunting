#!/bin/bash
#SBATCH -c 16
#SBATCH -p kempner
#SBATCH -t 1-12:00:00 
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus-per-node=1
#SBATCH --account=kempner_krajan_lab
#SBATCH --constraint=cc7.0|cc8.0|cc8.6
#SBATCH --exclude=holygpu8a[19604,19605,19606]
#SBATCH -J ma_zfish_cluster_eval
#SBATCH -o ./logs/MAZFish_cluster_eval/ma_zfish_cluster_eval_%j.out
#SBATCH -e ./logs/MAZFish_cluster_eval/ma_zfish_cluster_eval_%j.err


module load cuda/11.8.0-fasrc01
source activate zfish

#!/bin/bash
EVAL_ROLLOUT_THREADS=50
EVAL_EP_LENGTH=800
EVAL_RENDER_EPISODES=10
EXP_FOLDER_NAME=check
RUN_FOLDER=$1
RUN_DIR=$RUN_FOLDER

COMMON_ARGS="--additional_exps \
             --n_rollout_threads $EVAL_ROLLOUT_THREADS \
             --eval_episode_length $EVAL_EP_LENGTH \
             --eval_render_episodes $EVAL_RENDER_EPISODES \
             --eval_eating_distribution_decay=5 \
             --eval_uniform_reset_food_density=0.003 \
             --eval_uniform_max_food_density=0.003"

declare -A EXPS
EXPS["control"]="--additional_exps_name=control"
EXPS["food_speed"]="--additional_exps_name=food_speed --randomize_food_speed --min_random=0.0 --max_random=0.5"
EXPS["food_density"]="--additional_exps_name=food_density --randomize_food_density --min_random=0.001 --max_random=0.009"
EXPS["limit_convergence"]="--additional_exps_name=limit_convergence --eval_limit_convergence"
EXPS["limit_divergence"]="--additional_exps_name=limit_divergence --eval_limit_divergence"
# # # EXPS["food_speed_2"]="--additional_exps_name=food_speed_2 --eval_food_speed=2"
# # # EXPS["food_speed_0.5"]="--additional_exps_name=food_speed_0.5 --eval_food_speed=0.5"
# # # EXPS["food_speed_0"]="--additional_exps_name=food_speed_0 --eval_food_speed=0"
# # EXPS["food_density_0.002"]="--additional_exps_name=food_density_0.002 \
# #                               --eval_uniform_reset_food_density=0.002 \
# #                               --eval_uniform_max_food_density=0.003"
# # EXPS["food_density_0.005"]="--additional_exps_name=food_density_0.005 \
# #                               --eval_uniform_reset_food_density=0.005 \
# #                               --eval_uniform_max_food_density=0.006"
# # EXPS["food_density_0.004"]="--additional_exps_name=food_density_0.004 \
# #                               --eval_uniform_reset_food_density=0.004 \
# #                               --eval_uniform_max_food_density=0.005"
# # EXPS["food_density_0.001"]="--additional_exps_name=food_density_0.001 \
# #                               --eval_uniform_reset_food_density=0.001 \
# #                               --eval_uniform_max_food_density=0.002"
# EXPS["vergence_limit_divergence"]="--additional_exps_name=vergence_limit_divergence \
#                               --eval_max_left_vergence=-1.243 \
#                               --eval_max_right_vergence=1.243 \
#                               --eval_min_left_vergence=-1.243 \
#                               --eval_min_right_vergence=1.243"
# EXPS["vergence_limit_convergence"]="--additional_exps_name=vergence_limit_convergence \
#                               --eval_max_left_vergence=-0.756 \
#                               --eval_max_right_vergence=0.756 \
#                               --eval_min_left_vergence=-0.756 \
#                               --eval_min_right_vergence=0.756"
# EXPS["vergence_limit_mid"]="--additional_exps_name=vergence_limit_mid \
#                                 --eval_max_left_vergence=-0.999 \
#                                 --eval_max_right_vergence=0.999 \
#                                 --eval_min_left_vergence=-0.999 \
#                                 --eval_min_right_vergence=0.999"
# EXPS["vergence_limit_slight_converge"]="--additional_exps_name=vergence_limit_slight_converge \
#                                 --eval_max_left_vergence=-0.999 \
#                                 --eval_max_right_vergence=0.999 \
#                                 --eval_min_left_vergence=-1.243 \
#                                 --eval_min_right_vergence=1.243"
# EXPS["vergence_limit_slight_diverge"]="--additional_exps_name=vergence_limit_slight_diverge \
#                                 --eval_max_left_vergence=-0.756 \
#                                 --eval_max_right_vergence=0.756 \
#                                 --eval_min_left_vergence=-0.999 \
#                                 --eval_min_right_vergence=0.999"

# Run evals
for name in "${!EXPS[@]}"; do
    echo ">>> Running experiment: $name"
    python eval_ZFish.py "$RUN_DIR" $COMMON_ARGS ${EXPS[$name]}
    python preprocess_flatten.py "$RUN_DIR/outputs/additional_exps/$name" --additional_exps --delete_raw
done

python report_virtual_exps.py $RUN_DIR/outputs
