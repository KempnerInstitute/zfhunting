# Configurations/constants
# All distances in centimeters

import numpy as np
import matplotlib.cm as cm

m_to_cm = 100.0  # Conversion factor from meters to centimeters
cm_to_m = 1 / m_to_cm  # Conversion factor from centimeters to meters
deg2rad = np.pi / 180.0  # Conversion factor from degrees to radians
rad2deg = 1 / deg2rad  # Conversion factor from radians to degrees

# Fish
FISH_CONSTANTS = {
    "max_speed": 5,  # 30 mm/s when escaping, but around 20 mm/s when foraging
    "max_turn_speed": 7,  # 10 rad/s
    # "max_turn_speed": 3.5,  # trying half 
    "max_eye_turn_speed": 0.8,  # 4, 14 rad/s
    "perception_field": 163 * deg2rad,  # radians
    "max_left_vergence": -43.3
    * deg2rad,  # the angle of left eye at max convergence, radians
    "max_right_vergence": 43.3
    * deg2rad,  # the angle of right eye at max convergence, radians
    "min_left_vergence": -71.2
    * deg2rad,  # the angle of left eye at max divergence, radians
    "min_right_vergence": 71.2
    * deg2rad,  # 63.5 * deg2rad, # the angle of right eye at max divergence, radians
    "max_interbout_interval": 1,  # NOT USED unless AGENT_PARAMS["fatigue"] set to true, seconds, see Johnson et al 2020
    "bout_length": 0.2,  # NOT USED, seconds, see Johnson et al 2020 and Bolton et al 2019
    "eye_separation": 2,  # Distance between the eyes, mm (should be 0.5 mm)
    "eye_forward_offset": 0.5,  # Forward offset of the eyes, mm
}
FISH_CONSTANTS["eye_hypotenuse"] = np.linalg.norm(
    [FISH_CONSTANTS["eye_separation"] / 2, FISH_CONSTANTS["eye_forward_offset"]]
)  # Distance from center of head to eye, mm


# Object types
OBJECT_TYPES = {
    "NONE": 0.0,
    "FOOD": 1,
    "WALL": 0.5,
    "AGENT": -1,
    # "AGENT_MIN": 0.5,
    # "AGENT_MAX": 0.75,
}


# Environment parameters
ENV_PARAMS = {
    "food_radius": 0.075,  # radius of paramecium, mm
    "food_speed": 1,  # speed of paramecium, mm/s
    "food_speed_step": 0.001,  # updated 2025.09.17 11:27, 1/10 of previous value
    "food_turn_std": 10 * deg2rad,  # 6*deg2rad, # radians
    "food_turn_std_step": 0.1 * deg2rad * 0.1,  # updated 2025.09.17 11:27, 1/10 of previous value
    "arena_size_max": (
        100,
        100,
    ),  # (200, 200), # (300, 300), # mm, used in Johnson et al 2020 (Probabilistic models of larval zebrafish)
    "arena_size_min": (33, 33),
    "fps_video": 8,  # 1/FISH_CONSTANTS["bout_length"],  # 1 bout per frame
    "fps_sim": 8,  # 1/FISH_CONSTANTS["bout_length"], # 1 bout per frame
    "render_figsize": (16, 8),
    "max_food": 80,  # Maximum number of food items in the arena
    "max_patches": 4,  # Maximum number of patches in the arena
    # Added density parameters
    "reset_food_density": 0.015,  # NOT USED, only for consistency with arena.py
    "step_food_density": 0.001,  # NOT USED, only for consistency with arena.py
    "step_food_decay": 0.0,  # NOT USED, only for consistency with arena.py
    "reset_patch_density": 0.002,  # NOT USED, only for consistency with arena.py 0.001
    "step_patch_density": 0.00000,  # NOT USED, only for consistency with arena.py 0.000001
    "max_food_density": 0.02,  # NOT USED, only for consistency with arena.py
    "stockpile_density": 0.5,  # NOT USED, only for consistency with arena.py
    "uniform_max_food_density": 0.015,  # 0.004, 0.04, 0.005, #0.005
    "uniform_reset_food_density": 0.003,  # The initial food density at the start of training
    "uniform_step_food_density": 0.0,  # 0.00001,
    "min_food_density": 0.003,  # Final food density over training curriculum
    "patchy_max_food_density": 0.015,  # 0.002
    "patchy_reset_food_density": 0.015,
    "patchy_step_food_density": 0.001,
    "feeder_max_food_density": 0.06,
    "feeder_reset_food_density": 0.04,
    "pfeeder": 0,  # 1 / 6,  # Proportion of patchy-feeder arena
    "prandom": 0,  # 1 / 6,  # Proportion of patchy-r
    "urandom": 1,  # 4 / 6,  # Proportion of uniform-random arena
    "max_episode_length": 1200,  # Maximum number of steps in an episode
    "walker_agent_speed_ratio": 0.5,  # Ratio of walkerbot max speed to agent max speed
    "walker_agent_turn_ratio": 0.5,  # Ratio of walkerbot turn speed to agent turn speed
    "walker_agent_perception_radius": 10,  # Perception radius of walkerbot seeing agents, mm
}

FISH_CONSTANTS["max_step"] = (
    FISH_CONSTANTS["max_speed"] / ENV_PARAMS["fps_sim"]
)  # Convert speed to step size
FISH_CONSTANTS["max_turn"] = (
    FISH_CONSTANTS["max_turn_speed"] / ENV_PARAMS["fps_sim"]
)  # Convert turn speed to step angle
FISH_CONSTANTS["max_eye_turn"] = (
    FISH_CONSTANTS["max_eye_turn_speed"] / ENV_PARAMS["fps_sim"]
)  # Convert eye turn speed to step angle
ENV_PARAMS["max_step"] = FISH_CONSTANTS["max_step"]
ENV_PARAMS["max_turn"] = FISH_CONSTANTS["max_turn"]
ENV_PARAMS["walker_max_step"] = (
    ENV_PARAMS["walker_agent_speed_ratio"] * ENV_PARAMS["max_step"]
)
ENV_PARAMS["walker_max_turn"] = (
    ENV_PARAMS["walker_agent_turn_ratio"] * ENV_PARAMS["max_turn"]
)

ENV_PARAMS["food_speed"] = (
    ENV_PARAMS["food_speed"] / ENV_PARAMS["fps_sim"]
)  # Convert food drift speed to step size
ENV_PARAMS["food_speed_step"] = ENV_PARAMS["food_speed_step"] / ENV_PARAMS["fps_sim"]
FISH_CONSTANTS["max_interbout_interval"] = (
    FISH_CONSTANTS["max_interbout_interval"] * ENV_PARAMS["fps_sim"]
)  # Convert seconds to steps

# Agent parameters
AGENT_PARAMS = {
    # Sensing ranges
    "food_detection_range": 10,  # real is 2 mm?
    "fish_detection_range": 10,  # 50, # mm
    "sensing_radius": 10,  # (wall sensing radius + ray length), mm
    # Sensor params
    "num_rays": 10,
    "eating_angle": 80 * deg2rad,  # the angle of the eating cone, radians
    "baseline_success_eating_angle": 10 * deg2rad,  # Max error angle for perfect eating, also used for normalizing. Set to 0 to disable normalization.
    "eating_distribution_decay_step": 0.005,  # Higher is more decay  # updated 2025.09.17 11:27, 1/10 of previous value
    "eating_distribution_decay_start": 0.5,  # The initial value of eating distribution decay
    "eating_distribution_decay_max": 5,  # 10, # The maximum value of eating distribution decay  #  updated 2025.09.17 11:27, 1/2 of previous value
    "train_food_scaling_min": 0.25,  # Minimum scaling factor for food during training
    "train_food_scaling_max": 1.0,  # Max scaling factor for food during training
    "eating_radius": 1,  # 0.5, 1.5, # distance at which food is eaten
    "body_radius": 0.5,  # size of body, mm
    "distance_noise_std": 0.01,  # 0.05,  # Standard deviation of noise added to distance measurement
    "detection_failure_rate": 0.0,  # 0.1,  # Probability of complete detection failure
    "false_positive_rate": 0.0,
    # Reward/Internal variable params
    "energy_minimum": 0,  # NOT USED
    "energy_maximum": 100,  # NOT USED
    # "collective_sensing_radius": 15,
    # "collective_sensing_radius": 10,  # temp HACK
    "energy_per_step": 1,  # NOT USED
    "energy_food": 10,  # NOT USED
    # "fatigue_min": 0,
    # "fatigue_max": 1,
    # "fatigue_per_step": 1 / (ENV_PARAMS["fps_sim"] * 3),
    # "fatigue_recovery_per_step": 2 / ENV_PARAMS["fps_sim"],
    "max_food_eaten_per_step": 1,
    "discrete_actions": False,  # Whether to use discrete actions or continuous actions for move and turn
    "num_speeds": 5,  # Number of discrete speeds for the agent, only used if discrete_actions is True
    "num_turn_angles": 9,  # Number of discrete turn angles for the agent, only used if discrete_actions is True
    "penalize_move_threshold": 0.3,  # 0.3,  # Threshold for penalizing movement (value in 0 to 1), set to None to disable
    "penalize_turn_threshold": 0.3,  # 0.3,  # normalized |turn| threshold in [-1,1]
    "fatigue": False,
    "fatigue_recovery": 1
    / (
        FISH_CONSTANTS["max_interbout_interval"] * ENV_PARAMS["fps_sim"] / 2
    ),  # fatigue recovery per step
    "eye_fatigue_recovery": 2
    * (FISH_CONSTANTS["min_right_vergence"] - FISH_CONSTANTS["max_right_vergence"])
    / 20,
    "forced_interbout": None,
    "action_noise_std": 0.0,
    "use_1dof_eyes": True,
    "blur_effect": False,
    "eye_persistence": 0,  # 5
    "flash_monocular_only": False,  # Whether to flash only in monocular region
    "eye_muscle_model": False,  # Whether to use eye muscle model for vergence control
    "k_relax_eye": 0.1,  # Relaxation constant for eye
    "g_input_eye": 0.1,  # Gain for eye input
}

REWARDS = {
    "eat": 25,
    # "eat": 100,
    "collision": -5,
    "shape_reward": True,
    "large_move": -0.01,  # -0.05,  # Linear penalty for moving more than the threshold
    "large_turn": -0.01,  # -0.05,  # Linear penalty for moving more than the threshold
    "fatigue": -0.5,  # Penalty for fatigue
    "vergence_deviation": 0.0,  # -0.01 #-0.01 #-0.1, # Penalty for deviation from resting vergence
    "switching_vergence": 0.0,  # Penalty for switching vergence
    "max_align_reward": 0.0,
    "eye_fatigue_penalty": 0.0,
}

# Rendering color configuration
COLORS = {
    # "agent_bitten": "red",  # when agent is bitten
    # "agent_biting": "black",  # when agent is biting
    # "eating_other": "red",  # eating circle color when agent is biting
    "eating_food": "black",  # eating circle color when agent is eating
    "agent_eating_radius": "black",  # Eating radius indicator color -- unused?
    "food": "green",  # Food color
    "ray_default": "lightgray",  # Default ray color
    "ray_intersecting_food": "red",  # Ray color when intersecting with food
    "ray_intersecting_agent": "yellow",  # Ray color when intersecting with another agent
    "ray_observing_empty": "black",  # Ray color when observing nothing
    "ray_contrast": "purple",  # Ray color when observing nothing
    "knollen": "black",  # Line color for knollen sensing visualization
    "eating_cone": "cyan",  # Eating cone color
    "ampullary": "orange",  # Line color for ampullary sensing visualization
    "bounded_walk_circ": "blue",  # Bounded walk circumference color
}

SCALING_FACTORS = {
    "standard": 1.0,
    "half": 0.5,
    "double": 2.0,
    "quarter": 0.25,
    "ten_times": 10.0,
}

############### FEATURES ###############
# Custom colormap
twilight = cm.get_cmap("twilight", 256)
twilight_rotated = np.roll(
    twilight(np.linspace(0, 1, 256)), 64, axis=0
)  # Roll it by 64/256 so that 0 is no longer grey
twilight_rotated = cm.colors.ListedColormap(twilight_rotated)



FEATURE_METADATA = {
    # Circular features
    "orientation": {
        "color_mode": "circular",
        "feature_type": "circular",
        "name": "Agent Orientation",
        "short_name": r"$\theta$",
    },
}

OTHER_METADATA = {
    "auc_difference": {
        "name": "AUC Difference",
        "short_name": r"$\Delta$ AUC",
        "description": "Difference between average AUC for successful and non-tracking episodes",
    },
    "avg_auc_success": {
        "name": "Average AUC (Successful)",
        "short_name": "AUC (Success)",
        "description": "Average AUC for successful episodes",
    },
    "avg_auc_nontracking": {
        "name": "Average AUC (Non-Tracking)",
        "short_name": "AUC (Non-Tracking)",
        "description": "Average AUC for non-tracking episodes",
    },
    "eating_events_per_episode": {
        "name": "Eating Events per Episode",
        "short_name": "Eating Events",
        "description": "Average number of eating events per episode",
    },
    "vd": {
        "name": "Vergence Deviation (deg)",
        "short_name": "VD (deg)",
        "description": "Deviation from resting vergence (degrees)",
    },
    "large_move_penalty": {
        "name": "Large Move Penalty",
        "short_name": "Large Move Penalty",
        "description": "Penalty for large movements",
    },
    "large_turn_penalty": {
        "name": "Large Turn Penalty",
        "short_name": "Large Turn Penalty",
        "description": "Penalty for large turns",
    },
    "walkerbots": {
        "name": "Number of Bots",
        "short_name": "Num Bots",
        "description": "Number of aggressive bots",
    },
}
