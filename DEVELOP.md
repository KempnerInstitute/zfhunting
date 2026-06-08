# Development stuff

## List of key cfg features

(Note: most of these can be changed when calling training script (or in `sweep_configs.py` for running multiple trains/evals in parallel) via flags, but not all.)

* Curriculum learning: food speed, food turning angle, food density, and eating distribution decay are all increased through training for better convergence:
    * `ENV_PARAMS["food_speed"]` is the *final* food speed, and `ENV_PARAMS["food_speed_step"]` is the value the speed is increased by each training step (per rollout thread). The food speed always starts at 0.
    * Likewise, `ENV_PARAMS["food_turn_std"]` is the *final* food turn standard deviation, and `ENV_PARAMS["food_turn_std_step"]` is the value the food turn standard deviation is increased by each training step (per rollout thread). The food turn standard deviation also always starts at 0.
    * `ENV_PARAMS["uniform_reset_food_density"]` and `ENV_PARAMS["uniform_max_food_density"]` are the *initial* reset and max food densities at the start of training. Density decays exponentially with a randomly chosen multiplicative decay at each step between `all_args["train_food_scaling_min"]` and `all_args["train_food_scaling_max"]`. `train_food_scaling_max` always defaults to 1, but `train_food_scaling_min` can be passed as a flag in training. The *final* food density is `ENV_PARAMS["min_food_density"]`.
    * The eating distribution decay (Laplace distribution) starts at `AGENT_PARAMS["eating_distribution_decay_start"]`, increases by `AGENT_PARAMS["eating_distribution_decay_step"]` each training step, and reaches a maximum of `AGENT_PARAMS["eating_distribution_decay_max"]`.

* Food detection range vs. fish detection range vs. sensing radius:
    * `AGENT_PARAMS["food_detection_range"]` is the distance at which food is detected, `AGENT_PARAMS["fish_detection_range"]` is the distance at which fish is detected, and `AGENT_PARAMS["sensing_radius"]` is the distance at which walls are detected and what is shown in rendered episodes at the outermost detection range.

* FPS and time/step conversions:
    * `FISH_CONSTANTS` lists constants in terms of time, they are converted to step by dividing by the FPS (note the difference between `FISH_CONSTANTS["max_speed"]` and `FISH_CONSTANTS["max_step"]`)

* Eating angle and probability distribution
    * `AGENT_PARAMS["eating_angle"]` stores the hard cutoff for eating, but within the valid range there is a probabality distribution for eating based on the angle to the food - it's a Laplace distribution with decay given by `AGENT_PARAMS["eating_distribution_decay_max"]` (note the curriculum learning above)

* Perception noise
    * `AGENT_PARAMS["distance_noise_std"]` is the standard deviation of the uniform multiplicative noise added to distance measurements.
    * `AGENT_PARAMS["detection_failure_rate"]` is the probability a single sensor (in either eye, not limited to monocular/binocular) will fail to detect everything (false negative), and receive no object instead
    * `AGENT_PARAMS["false_positive_rate"]` is the probability a single sensor (in either eye, not limited to monocular/binocular) will detect food (false positive), regardless of what is actually there
    * Note that `AGENT_PARAMS["perception_type"]` is set to `"rays"` in `cfg.py` but is always overrided with `"projected"` via a flag when training

* Discrete actions
    * If `AGENT_PARAMS["discrete_actions"]` is set to `True`, `AGENT_PARAMS["num_speeds"]` and `AGENT_PARAMS["num_turn_angles"]` give the number of discrete speeds and turn angles (forward speed is logarithmically spaced, turn angle is linearly spaced).

* Movement/eye fatigue model
    * Movement fatigue was implemented to generate bouts and interbout intervals, but failed: if `AGENT_PARAMS["fatigue"]` is set to `True`, the agent uses an internal count of the number of steps where its forward speed has been positive. If the forward speed is 0, this count decreases by `AGENT_PARAMS["fatigue_recovery"]` each step (which is defined in terms of `FISH_CONSTANTS["max_interbout_interval"]`). All fatigue is penalized by multiplying the fatigue count with `REWARDS["fatigue"]`. This hypothetically should force the agent to decide to stop and let fatigue decrease or keep moving at the expense of higher fatigue. Alternatively, if `AGENT_PARAMS["forced_interbout"]` is set to an integer `n` (not `None`), then the agent can only make a positive forward speed action after every `n` steps.
    * Eye fatigue is similar but aims to increase persistence of vergence angle. Changes in eye angle are added to an internal `eye_fatigue` variable. If `all_args["eye_fatigue_penalty"]` is set to a nonzero value, then when eye fatigue surpasses a threshold, there is a ReLU like penalty. Otherwise, there is no reward penalty but the agent is constrained such that it cannot further move its eyes if above the threshold and must wait before eye fatigue decreases again. The threshold is the max eye angle change from diverged to converged. The linear decrease in eye fatigue each step is `AGENT_PARAMS["eye_fatigue_recovery"]`. Note that `"eye_fatigue_penalty"` is not contained in `cfg.py` but rather passed as a flag when training, otherwise it defaults to 0 and the constrained model is used.

* Other misc features
    * `AGENT_PARAMS["penalize_move_threshold"]` is the move forward threshold above which there is a ReLU-like penalty. Should be between 0 and 1, set to `None` to disable.
    * `AGENT_PARAMS["action_noise_std"]` is the standard deviation of the uniform multiplicative action noise independently added to move and turn actions (not eyes).
    * `AGENT_PARAMS["use_1dof_eyes"]` strings together the eyes such that instead of independently controlling each eye, the agent only controls 1 eye and the other is always at the reflected angle.
    * If `binary_eye_state` is passed as a flag during training, then the eyes can only be either converged or diverged at all times. Therefore, if `AGENT_PARAMS["use_1dof_eyes"]` is also `True`, there are 2 eye states, otherwise there are 4 eye states. Note that `"binary_eye_state"` is not contained in `cfg.py` but rather passed as a flag when training.
    * `AGENT_PARAMS["blur_effect"]` only works with `binary_eye_state` being on. If it is set to `True`, the agent receives no sensory observations after switching the state of one of the eyes.
    * `AGENT_PARAMS["eye_persistence"]` also only works with `binary_eye_state` being on. If it is set to a positive integer `n`, then the agent can only make a change to the eye state after every `n` steps.
    * `AGENT_PARAMS["use_separate_eyes"]` controls whether eyes should have the same position or different positions (creating the blind spot). It was kept for backward compatability with the initial zebrafish model, but likely will have bugs if this is set to `False`.
    * If `AGENT_PARAMS["flash_monocular_only"]` is set to `True`, observations in the monocular region are flashes -- any object there is observed for one step but then set to nothing. The idea for this was to strongly encourage converging as an extreme case.
    * If `AGENT_PARAMS["eye_muscle_model"]` is `True`, then eye change actions become "driver forces" instead of angular changes -- the eye now follows muscle dynamics where the action is how much the eye flexes toward convergence. Only in this case `AGENT_PARAMS["k_relax_eye"]` and `AGENT_PARAMS["g_input_eye"]` are used, which are the relaxation constant and the gain on the eye input force, respectively.

* Rewards
    * `REWARDS["eat"]` is the reward (positive) for successfully eating food.
    * `REWARDS["collision"]` is the reward (negative) for colliding with other agents, not used if only single agent.
    * `REWARDS["shape_reward"]` can be set to `True` or `False` for distance-based reward shaping. Note that distance-based reward is equal to the proportion of the arena covered in a single step in terms of getting closer to all food, so there is no parameter to tune.
    * `REWARDS["large_move"]` is only used if `AGENT_PARAMS["penalize_move_threshold"]` is not `None` and it is the slope of the ReLU-like move forward penalty.
    * `REWARDS["fatigue"]` is only used if `AGENT_PARAMS["fatigue"]` is set to `True` and is multiplied with the fatigue count of the agent for computing the fatigue penalty.
    * `REWARDS["vergence_deviation"]` is multiplied by the deviation of each eye angle from resting position (max divergence) for computing the vergence deviation penalty.
    * `REWARDS["switching_vergence"]` is only used if `binary_eye_state` (see above) is on, and is the penalty per step if the eye state changes (per eye).
    * `REWARDS["max_align_reward"]` is used for orientation-based reward shaping, and orientation reward is computed via `cos(angle_diff) * REWARDS["max_align_reward"] * (1 - nearest_food_distance / food_sensing_radius)` where `angle_diff` is the angle to the nearest food.
