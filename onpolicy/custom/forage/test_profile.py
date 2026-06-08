import cProfile, pstats
from MAZFish import main
import cfg
from datetime import datetime

def profile_run(all_args):
    prof = cProfile.Profile()
    prof.enable()
    main(all_args)
    prof.disable()
    stats = pstats.Stats(prof).sort_stats("cumtime")  # sort by total time
    stats.print_stats(20)  # show top 20

if __name__ == "__main__":
    all_args = {
        "num_agents": 1,
        "arena_size": (100, 100),
        "arena_shape": "circle",
        "reset_food_density": cfg.ENV_PARAMS["reset_food_density"], # 0.015,
    "step_food_density": cfg.ENV_PARAMS["step_food_density"], # 0.00001,
    "step_food_decay": cfg.ENV_PARAMS["step_food_decay"], # 0.00001,
    "reset_patch_density": cfg.ENV_PARAMS["reset_patch_density"], # 0.00001,
    "step_patch_density": cfg.ENV_PARAMS["step_patch_density"], # 0.00001,
    "max_food_density": cfg.ENV_PARAMS["max_food_density"], # 0.02, #0.02
    "stockpile_density": cfg.ENV_PARAMS["stockpile_density"],
    "uniform_max_food_density": cfg.ENV_PARAMS["uniform_max_food_density"],
    "uniform_reset_food_density": cfg.ENV_PARAMS["uniform_reset_food_density"],
    "uniform_step_food_density": cfg.ENV_PARAMS["uniform_step_food_density"],
    "patchy_max_food_density": cfg.ENV_PARAMS["patchy_max_food_density"],
    "patchy_reset_food_density": cfg.ENV_PARAMS["patchy_reset_food_density"],
    "feeder_max_food_density": cfg.ENV_PARAMS["feeder_max_food_density"],
    "feeder_reset_food_density": cfg.ENV_PARAMS["feeder_reset_food_density"],
        "energy_food": cfg.AGENT_PARAMS["energy_food"],
        # "render_mode": None,
        "render_mode": "rgb_array",
        "sensing_radius": 20,
        "timestamp" : datetime.now().strftime("%Y%m%d_%H%M%S"),
        "max_food_eaten_per_step": 1,
        "max_episode_length": 1200,
        "perception_type": "projected", # "rays"
        "pfeeder": 0,
        "prandom": 0,
        "urandom": 1,
        "p_init_closeby": 0.0,
        "is_eval": False,
        "discrete_actions": False,
        "use_separate_eyes": True,
        "use_1dof_eyes": True,
        "binocular_depth_only": True,
        "binary_eye_state": True,
        "eye_persistence": 1,
        "blur_effect": False,
        "vergence_deviation": 0.0,
    }
    profile_run(all_args)
