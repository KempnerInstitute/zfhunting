import sys
import os
import socket
from datetime import datetime
import setproctitle
import numpy as np
from pathlib import Path
import torch
import random
import ast
import string
from onpolicy.config import get_config
from onpolicy.custom.forage import cfg
from onpolicy.envs.env_wrappers import SubprocVecEnv, DummyVecEnv
import time  # Add the time module for timing

from onpolicy.custom.forage.MAZFish import MultiAgentZFishEnv

from onpolicy.custom.forage.ZFish_runner_shared import MAZFishRunner as SharedRunner
from onpolicy.custom.forage.ZFish_runner_separated import MAZFishRunner as SeparatedRunner
from pprint import pprint
from utils_logging import log_on_done


def tuple_of_ints(value):
    """
    Safely parse a Python tuple literal or a comma‐separated string of ints.
    Examples:
      "(70,70)"  → (70, 70)
      "70,70"    → (70, 70)
    """
    if value.lower() == "none":
        return None
    try:
        # first try a real Python literal
        t = ast.literal_eval(value)
        if isinstance(t, tuple) and all(isinstance(x, int) for x in t):
            return t
    except (ValueError, SyntaxError):
        pass
    # fallback: split on commas
    parts = value.strip("()[] ").split(",")
    return tuple(int(p) for p in parts)


def make_env(all_args, eval=False):
    def get_env_fn(rank):
        def init_env():
            env_seed = all_args.seed * 50000 + rank * 10000 if eval else all_args.seed + rank * 1000
            if not eval:
                env = MultiAgentZFishEnv(vars(all_args), seed=env_seed, done_callback=log_on_done, is_eval=False)
            else:
                # check for virtual experiments
                random.seed(all_args.seed * 50000 + rank * 10000 + 12345)  # ensure different across eval envs
                if getattr(all_args, "randomize_food_speed", False):
                    # print("Randomizing food speed for eval env")
                    all_args.food_speed_start = random.uniform(
                        all_args.min_random, all_args.max_random
                    )
                    all_args.food_speed_max = all_args.food_speed_start
                    env_seed = all_args.seed * 50000 # same seed for all envs for experiment
                    random.seed(env_seed)
                    # print(f"Food speed set to {all_args.food_speed_start}")
                if getattr(all_args, "randomize_food_density", False):
                    all_args.uniform_reset_food_density = random.uniform(
                        all_args.min_random, all_args.max_random
                    )
                    all_args.uniform_max_food_density = all_args.uniform_reset_food_density
                    env_seed = all_args.seed * 50000 # same seed for all envs for experiment
                    random.seed(env_seed)
                if getattr(all_args, "eval_limit_convergence", False):
                    all_args.cfg_override["FISH_CONSTANTS"]["max_left_vergence"] = random.uniform(
                        cfg.FISH_CONSTANTS["min_left_vergence"],
                        cfg.FISH_CONSTANTS["max_left_vergence"],
                    )
                    all_args.cfg_override["FISH_CONSTANTS"]["max_right_vergence"] = -all_args.cfg_override["FISH_CONSTANTS"]["max_left_vergence"]
                    env_seed = all_args.seed * 50000 # same seed for all envs for experiment
                    random.seed(env_seed)
                if getattr(all_args, "eval_limit_divergence", False):
                    all_args.cfg_override["FISH_CONSTANTS"]["min_left_vergence"] = random.uniform(
                        cfg.FISH_CONSTANTS["min_left_vergence"],
                        cfg.FISH_CONSTANTS["max_left_vergence"],
                    )
                    all_args.cfg_override["FISH_CONSTANTS"]["min_right_vergence"] = -all_args.cfg_override["FISH_CONSTANTS"]["min_left_vergence"]
                    env_seed = all_args.seed * 50000 # same seed for all envs for experiment
                    random.seed(env_seed)
                if getattr(all_args, "randomize_num_bots", False):
                    all_args.num_walkerbots = rank
                    env_seed = all_args.seed * 50000 # same seed for all envs for experiment
                    random.seed(env_seed)
                    
                env = MultiAgentZFishEnv(vars(all_args), seed=env_seed, is_eval=True)
            env.seed(env_seed)
            return env
        return init_env

    num_threads = all_args.n_eval_rollout_threads if eval else all_args.n_rollout_threads
    print(f"Making {num_threads} {'Eval' if eval else 'Train'} environments...")
    
    if num_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(num_threads)])


def parse_args(args, parser):
    parser.add_argument("--num_agents", type=int, default=4)
    parser.add_argument("--max_food", type=int, default=cfg.ENV_PARAMS["max_food"])
    # parser.add_argument("--num_patches", type=int, default=cfg.ENV_PARAMS["max_patches"])
    parser.add_argument("--discrete_actions", type=bool, default=cfg.AGENT_PARAMS["discrete_actions"])
    parser.add_argument("--reset_food_density", type=float, default=cfg.ENV_PARAMS["reset_food_density"])
    parser.add_argument("--step_food_density", type=float, default=cfg.ENV_PARAMS["step_food_density"])
    parser.add_argument("--step_food_decay", type=float, default=cfg.ENV_PARAMS["step_food_decay"])
    parser.add_argument("--reset_patch_density", type=float, default=cfg.ENV_PARAMS["reset_patch_density"])
    parser.add_argument("--step_patch_density", type=float, default=cfg.ENV_PARAMS["step_patch_density"])
    parser.add_argument("--max_food_density", type=float, default=cfg.ENV_PARAMS["max_food_density"])
    parser.add_argument("--stockpile_density", type=float, default=cfg.ENV_PARAMS["stockpile_density"])
    parser.add_argument("--uniform_max_food_density", type=float, default=cfg.ENV_PARAMS["uniform_max_food_density"])
    parser.add_argument("--uniform_reset_food_density", type=float, default=cfg.ENV_PARAMS["uniform_reset_food_density"])
    parser.add_argument("--uniform_step_food_density", type=float, default=cfg.ENV_PARAMS["uniform_step_food_density"])
    parser.add_argument("--patchy_max_food_density", type=float, default=cfg.ENV_PARAMS["patchy_max_food_density"])
    parser.add_argument("--patchy_reset_food_density", type=float, default=cfg.ENV_PARAMS["patchy_reset_food_density"])
    parser.add_argument("--patchy_step_food_density", type=float, default=cfg.ENV_PARAMS["patchy_step_food_density"])
    parser.add_argument("--feeder_max_food_density", type=float, default=cfg.ENV_PARAMS["feeder_max_food_density"])
    parser.add_argument("--feeder_reset_food_density", type=float, default=cfg.ENV_PARAMS["feeder_reset_food_density"])
    parser.add_argument("--binocular_depth_only", action='store_true', default=False,)
    parser.add_argument("--binocular_angle_only", action='store_true', default=False,)
    parser.add_argument("--conspecific_monocular_perception", action='store_true', default=True,)
    parser.add_argument("--angle_noise_std_food", type=float, default=0.0, help="std of uniform noise added to food angles (in radians)")
    parser.add_argument("--angle_noise_std_walker", type=float, default=0.0, help="std of uniform noise added to walkerbot angles (in radians)")
    parser.add_argument("--binary_eye_state", action='store_true', default=False,)
    parser.add_argument("--use_1dof_eyes", action='store_true', default=False,)
    parser.add_argument("--vergence_deviation", type=float, default=cfg.REWARDS["vergence_deviation"])
    parser.add_argument("--flash_monocular_only", action='store_true', default=cfg.AGENT_PARAMS["flash_monocular_only"])
    parser.add_argument("--detection_failure_rate", type=float, default=cfg.AGENT_PARAMS["detection_failure_rate"])
    parser.add_argument("--false_positive_rate", type=float, default=cfg.AGENT_PARAMS["false_positive_rate"])
    parser.add_argument("--eye_muscle_model", action='store_true', default=cfg.AGENT_PARAMS["eye_muscle_model"])
    parser.add_argument("--k_relax_eye", type=float, default=cfg.AGENT_PARAMS["k_relax_eye"])
    parser.add_argument("--g_input_eye", type=float, default=cfg.AGENT_PARAMS["g_input_eye"])
    parser.add_argument("--eating_distribution_decay_start", type=float, default=cfg.AGENT_PARAMS["eating_distribution_decay_start"])
    parser.add_argument("--eating_distribution_decay_step", type=float, default=cfg.AGENT_PARAMS["eating_distribution_decay_step"])
    parser.add_argument("--eating_distribution_decay_max", type=float, default=cfg.AGENT_PARAMS["eating_distribution_decay_max"])
    parser.add_argument("--food_speed_start", type=float, default=0.0)
    parser.add_argument("--food_speed_max", type=float, default=cfg.ENV_PARAMS["food_speed"])
    parser.add_argument("--food_speed_step", type=float, default=cfg.ENV_PARAMS["food_speed_step"])
    parser.add_argument("--food_turn_std_start", type=float, default=0.0)
    parser.add_argument("--food_turn_std_max", type=float, default=cfg.ENV_PARAMS["food_turn_std"])
    parser.add_argument("--food_turn_std_step", type=float, default=cfg.ENV_PARAMS["food_turn_std_step"])
    parser.add_argument(
        "--num_vids_to_save",
        type=int,
        default=1,
        help="Number of vids to save during training",
    )
    parser.add_argument(
        "--pfeeder", type=float, default=cfg.ENV_PARAMS["pfeeder"]
    )  # proportion of patchy-feeder arena
    parser.add_argument(
        "--prandom", type=float, default=cfg.ENV_PARAMS["prandom"]  
    )  # proportion of patchy-random arena
    parser.add_argument(
        "--urandom", type=float, default=cfg.ENV_PARAMS["urandom"]
    )  # proportion of uniform-random arena
    parser.add_argument("--food_radius", type=float, default=cfg.ENV_PARAMS["food_radius"])
    parser.add_argument("--food_preset_mode", type=str, default=None)
    parser.add_argument("--render_mode", type=str, default="rgb_array")
    parser.add_argument("--shared_reward", action="store_true", default=False)
    parser.add_argument("--collective_sensing_mode", type=str, choices=[None, 'simple_extension'])
    parser.add_argument("--sensing_radius", type=int, default=cfg.AGENT_PARAMS["sensing_radius"])
    parser.add_argument("--max_gif_frames", type=int, default=500)
    parser.add_argument("--max_episode_length", type=int, default=cfg.ENV_PARAMS["max_episode_length"])
    parser.add_argument("--max_food_eaten_per_step", type=int, default=cfg.AGENT_PARAMS["max_food_eaten_per_step"])
    parser.add_argument("--food_detection_range", type=float, default=cfg.AGENT_PARAMS["food_detection_range"])
    parser.add_argument("--fish_detection_range", type=float, default=cfg.AGENT_PARAMS["fish_detection_range"])
    parser.add_argument("--eating_angle", type=float, default=cfg.AGENT_PARAMS["eating_angle"])
    parser.add_argument("--baseline_success_eating_angle", type=float, default=cfg.AGENT_PARAMS["baseline_success_eating_angle"])
    parser.add_argument("--action_noise_std", type=float, default=cfg.AGENT_PARAMS["action_noise_std"])
    parser.add_argument("--blur_effect", action='store_true', default=cfg.AGENT_PARAMS["blur_effect"])
    parser.add_argument("--eye_persistence", type=int, default=cfg.AGENT_PARAMS["eye_persistence"],)
    parser.add_argument("--use_eye_fatigue", action='store_true', default=False)
    parser.add_argument("--eye_fatigue_recovery", type=float, default=cfg.AGENT_PARAMS["eye_fatigue_recovery"])
    parser.add_argument("--eye_fatigue_penalty", type=float, default=cfg.REWARDS["eye_fatigue_penalty"])
    parser.add_argument("--train_food_scaling_min", type=float, default=cfg.AGENT_PARAMS["train_food_scaling_min"])
    parser.add_argument("--train_food_scaling_max", type=float, default=cfg.AGENT_PARAMS["train_food_scaling_max"])
    parser.add_argument(
        "--attn_mode",
        type=str,
        default=None,
        choices=["x", "hx", "x+hx", None],
        help="Attention input mode over observations: x, hx, x+hx, or None to disable.",
    )
    parser.add_argument(
        "--attn_use_softmax",
        action="store_true",
        default=False,
        help="Use softmax for attention gating instead of sigmoid",
    )

    parser.add_argument("--num_walkerbots", type=int, default=0)
    parser.add_argument("--walker_pursuit", action='store_true', default=False)
    parser.add_argument("--walker_ignore_walls", action='store_true', default=False, help='if true, disables wall avoidance for walkerbots')
    parser.add_argument("--timestamp", type=str, default=None, help="timestamp; if None, use current time")
    parser.add_argument("--results_parent_dir", type=str, default=None)
    parser.add_argument("--curriculum_type", type=str, default="fixed_step_with_max")
    parser.add_argument("--curriculum_early_end_frac", type=float, default=0.9)  # only used in the time_normalized_step curriculum
    parser.add_argument("--r_eat_override", type=float, default=None)  # Override the eating reward (for tuning)
    parser.add_argument("--r_collide_override", type=float, default=None)  # Override the collision reward (for tuning)

    parser.add_argument("--r_collide_start", type=float) #If none and r_collide_override is defined, use that value
    parser.add_argument("--r_collide_end", type=float, default=None)#If none and r_collide_override is defined, use that value
    parser.add_argument("--r_collide_step", type=float, default=None)#If none and r_collide_override is defined, use 0

    parser.add_argument("--large_move_penalty", type=float, default=cfg.REWARDS["large_move"])
    parser.add_argument("--large_turn_penalty", type=float, default=cfg.REWARDS["large_turn"])

    parser.add_argument("--arena_size_max", type=int, default=None)  # Override the max arena size (for tuning)
    parser.add_argument("--arena_size_min", type=int, default=None)  # Override the min arena size (for tuning)

    all_args = parser.parse_known_args(args)[0]
    return all_args


def setup_algorithm(all_args):
    if all_args.algorithm_name == "rmappo":
        print("Using rmappo, setting use_recurrent_policy to True")
        all_args.use_recurrent_policy = True
        all_args.use_naive_recurrent_policy = False
    elif all_args.algorithm_name == "mappo":
        print("Using mappo, setting use_recurrent_policy & use_naive_recurrent_policy to False")
        all_args.use_recurrent_policy = False
        all_args.use_naive_recurrent_policy = False
    elif all_args.algorithm_name == "ippo":
        print("Using ippo, setting use_centralized_V to False")
        all_args.use_centralized_V = False
    else:
        raise NotImplementedError


def setup_device(all_args):
    if all_args.cuda and torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using device:", device)
        torch.set_num_threads(all_args.n_training_threads)
        if all_args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        device = torch.device("cpu")
        print("Using device:", device, "Threads:", all_args.n_training_threads)
        torch.set_num_threads(all_args.n_training_threads)
    return device


def setup_run_dir(all_args):
    print("Setting up run dir...")
    results_parent_dir = all_args.results_parent_dir
    print("results_parent_dir:", results_parent_dir)
    if results_parent_dir is None:
        CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
        run_dir = (
            Path(CURRENT_DIR)
            / "results"
            / f"{all_args.algorithm_name}-{all_args.env_name}-{all_args.experiment_name}"
        )
    else:
        run_dir = (
            Path(results_parent_dir)
            / "results"
            / f"{all_args.algorithm_name}-{all_args.env_name}-{all_args.experiment_name}"
        )
    print("run_dir:", run_dir)
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    curr_run = f"{all_args.timestamp}_{all_args.num_agents}_{all_args.run_name}"

    run_dir = run_dir / curr_run  # append
    if not run_dir.exists():
        os.makedirs(str(run_dir))
    else:
        raise Exception(f"run_dir {run_dir} already exists! Be sure to use a unique name or delete the folder if you want to overwrite.")

    all_args.run_dir = str(run_dir)
    return run_dir

def update_cfg_args(all_args):
    all_args.cfg_override = {
        "AGENT_PARAMS": {},
        "ENV_PARAMS": {},
        "REWARDS": {},
        "OBJECT_TYPES": {},
        "FISH_CONSTANTS": {},
    }
    if all_args.r_eat_override is not None:
        all_args.cfg_override["REWARDS"]["eat"] = all_args.r_eat_override
    if all_args.r_collide_override is not None:
        all_args.cfg_override["REWARDS"]["collision"] = all_args.r_collide_override 


    if all_args.arena_size_max is not None:
        all_args.cfg_override["ENV_PARAMS"][
            "arena_size_max"
        ] = (all_args.arena_size_max, all_args.arena_size_max)
    if all_args.arena_size_min is not None:
        all_args.cfg_override["ENV_PARAMS"][
            "arena_size_min"
        ] = (all_args.arena_size_min, all_args.arena_size_min)
    return all_args


def main(args):
    start_time = time.time()  # Start timing the execution of the main function

    parser = get_config()
    all_args = parse_args(args, parser)
    all_args = update_cfg_args(all_args)

    # OVERRIDES
    all_args.env_name = "MultiAgentForagingEnv"
    all_args.algorithm_name = ["rmappo", "ippo"][0]
    all_args.n_rollout_threads = 10
    # all_args.num_env_steps = 10
    all_args.render_episodes = 1
    all_args.episode_length = 600
    all_args.max_episode_length = all_args.episode_length

    all_args.share_policy = True
    if all_args.timestamp is None:
        all_args.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print("shared_reward", all_args.shared_reward)
    # pprint(vars(all_args))
    # print(vars(all_args))
    setup_algorithm(all_args)
    device = setup_device(all_args)
    run_dir = setup_run_dir(all_args)
    print("hidden_size", all_args.hidden_size)

    setproctitle.setproctitle(
        f"{all_args.algorithm_name}-{all_args.env_name}-{all_args.experiment_name}@{all_args.user_name}"
    )

    torch.manual_seed(all_args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    env_setup_start_time = time.time()  # Start timing environment setup
    envs = make_env(all_args, eval=False)
    eval_envs = make_env(all_args, eval=True) if all_args.use_eval else None
    print(f"Environment setup took {time.time() - env_setup_start_time:.2f} seconds.")  # Log environment setup time

    num_agents = all_args.num_agents

    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "num_agents": num_agents,
        "device": device,
        "run_dir": run_dir,
    }

    training_start_time = time.time()  # Start timing training
    Runner = SharedRunner if all_args.share_policy else SeparatedRunner
    runner = Runner(config)
    runner.run()
    print(f"Training took {time.time() - training_start_time:.2f} seconds.")  # Log training time

    rendering_start_time = time.time()  # Start timing rendering
    print("Done training, now rendering....")
    runner.render()
    print(f"Rendering took {time.time() - rendering_start_time:.2f} seconds.")  # Log rendering time

    envs.close()
    if all_args.use_eval and eval_envs is not envs:
        eval_envs.close()

    runner.writter.export_scalars_to_json(str(runner.log_dir + "/summary.json"))
    runner.writter.close()

    total_time = time.time() - start_time  # Calculate total execution time
    print(f"Total execution time: {total_time:.2f} seconds.")


if __name__ == "__main__":
    main(sys.argv[1:])
