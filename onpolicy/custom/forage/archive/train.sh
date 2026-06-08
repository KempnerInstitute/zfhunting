NUM_ENV_STEPS=2000000
NUM_AGENTS=1
PERCEPTION_TYPE="projected"

python train_ZFish.py --num_env_steps=$NUM_ENV_STEPS \
                            --num_agents=$NUM_AGENTS \
                            --perception_type=$PERCEPTION_TYPE \
                            --binocular_depth_only \
                            --use_1dof_eyes \
                            --binary_eye_state

python report_batch_preprocessing.py

jupyter nbconvert report_behavior_shs.ipynb --to python; python -u report_behavior_shs.py

# python train_foraging.py --num_env_steps=$NUM_ENV_STEPS \
#                             --num_agents=$NUM_AGENTS \
#                             --perception_type=$PERCEPTION_TYPE \
#                             --binocular_depth_only=False \
#                             --use_1dof_eyes=$USE_1DOF_EYES

# python report_batch_preprocessing.py

# jupyter nbconvert report_behavior_shs.ipynb --to python; python -u report_behavior_shs.py