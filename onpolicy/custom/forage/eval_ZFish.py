"""
Helpful notes:

Code Flow:
1. Parse evaluation-specific command-line arguments `eval_[...]`.
2. Load saved JSONs for training arguments (`all_args`) and environment settings (`env_args`).
3. Override training arguments with evaluation-time options via `update_all_args()`.
4. Reconstruct the environment and agent configuration using `make_env`.
5. Instantiate runner and run `.render()`.
6. Log the configuration used in a CSV for easy reference.

To Add a new eval argument:
1. Add the `--eval_<argname>` to the argparsing.
2. In `update_all_args()`, check if `eval_args.eval_<argname>` is not None and apply it to `all_args`.
"""

import os
import argparse
import json
import glob
from pathlib import Path
from datetime import datetime
import ast
import random
import string

import numpy as np

from onpolicy.custom.forage.ZFish_runner_shared import MAZFishRunner as SharedRunner
from onpolicy.custom.forage.ZFish_runner_separated import MAZFishRunner as SeparatedRunner
from onpolicy.custom.forage.train_ZFish import (
    make_env,
    setup_device,
)  # maybe these should go in a utilities file
import cfg


def int_or_none(value):
    if value.lower() == "none":
        return None
    return int(value)


def float_or_none(value):
    if value.lower() == "none":
        return None
    return float(value)


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


def list_of_ints(value):
    """
    Safely parse a Python list literal or a comma-separated string of ints.

    Examples:
      "[70, 70]"  → [70, 70]
      "70, 70"    → [70, 70]
    """
    if value.lower() == "none":
        return None

    if isinstance(value, list):
        return [int(v) for v in value]

    try:
        # Try to parse as a Python literal
        parsed = ast.literal_eval(value)
        if isinstance(parsed, list):
            return [int(v) for v in parsed]
    except (ValueError, SyntaxError):
        pass  # Fall back to comma-splitting

    # Parse as a comma-separated string
    value = value.strip("()[] ")
    parts = [p.strip() for p in value.split(",") if p.strip()]
    return [int(p) for p in parts]


def read_args_from_file(log_dir, arg_str="all_args", return_dict=False):
    print(log_dir)
    args_filename = glob.glob(str(log_dir / f"*_{arg_str}.json"))[0]

    with open(args_filename, "r") as f:
        args_dict = json.load(f)

    if return_dict:
        return args_dict

    all_args = argparse.Namespace()
    all_args.__dict__.update(args_dict)
    return all_args


def get_config_vals_of_interest(all_args):
    """
    Generally, these will be the flags (largely from train_fish.py) that
    we might want to vary between train and test.
    Typically, no need to add flags that would break model if changed
    (e.g. things that change obs_dim, like the number of rays)

    No need to be comprehensive since metadata is stored in the pkl file,
    this is just for quick reference and easy comparison generation.
    """
    episode_config = {
        "pfeeder": all_args.pfeeder,
        "prandom": all_args.prandom,
        "urandom": all_args.urandom,
        "reset_food_density": all_args.reset_food_density,
        "step_food_density": all_args.step_food_density,
        "step_food_decay": all_args.step_food_decay,
        "max_food_density": all_args.max_food_density,
        "eating_distribution_decay": all_args.eating_distribution_decay_start,
        "eval_seed": all_args.seed,
        "food_speed": all_args.food_speed_start,
        "food_turn_std": all_args.food_turn_std_start,
    }
    # NOTE add more values at end as needed
    return episode_config


def log_exp_info(random_id, run_dir, all_args, log_dir):
    """
    Log the experiment info to a quick-reference csv file that associates
    pkl ids with the config values used for that run.
    """
    episode_config = get_config_vals_of_interest(all_args)
    current_time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir_name = run_dir.split("/")[-1]

    exp_info_filename = log_dir / f"{run_dir_name}_exp_info.csv"
    if not os.path.exists(exp_info_filename):
        with open(exp_info_filename, "w") as f:
            f.write("pkl_id\t")
            f.write("run_timestamp\t")
            f.write("to_process\t")
            f.write("\t".join(episode_config.keys()))
            f.write("\n")
    with open(exp_info_filename, "a") as f:
        f.write(f"{random_id}\t")
        f.write(f"{current_time_str}\t")
        f.write("1\t")
        f.write("\t".join([str(val) for val in episode_config.values()]))
        f.write("\n")


def get_old_cfg_args(env_args):
    """
    Get the values from the env_args file that are also in the cfg file
    to ensure compatibility between env params and trained model.
    """
    all_env_params = {}
    agent_params = {
        k: v for k, v in env_args["agent_env_args"].items() if k in cfg.AGENT_PARAMS
    }
    env_params = {
        k: v for k, v in env_args["multi_agent_env_args"].items() if k in cfg.ENV_PARAMS
    }
    object_types = env_args["multi_agent_env_args"]["OBJECT_TYPES"]

    all_env_params["AGENT_PARAMS"] = agent_params
    all_env_params["ENV_PARAMS"] = env_params
    all_env_params["OBJECT_TYPES"] = object_types

    if "reward_params" in env_args["multi_agent_env_args"]:
        all_env_params["REWARDS"] = env_args["multi_agent_env_args"]["reward_params"]

    if "FISH_CONSTANTS" in env_args["multi_agent_env_args"]:
        all_env_params["FISH_CONSTANTS"] = env_args["multi_agent_env_args"][
            "FISH_CONSTANTS"
        ]

    return all_env_params


def update_all_args(all_args, eval_args):
    """
    Update all_args with the values from eval_args (if not None).
    """
    all_args.save_vids = eval_args.save_vids
    all_args.episode_length = eval_args.eval_episode_length
    all_args.render_episodes = eval_args.eval_render_episodes
    all_args.n_eval_rollout_threads = eval_args.n_rollout_threads
    all_args.n_rollout_threads = eval_args.n_rollout_threads
    all_args.num_vids_to_save = eval_args.num_vids_to_save
    all_args.actor_load_path = eval_args.eval_actor_load_path #always expect an actor load path: if none, use default.

    if eval_args.eval_num_agents is not None:
        all_args.num_agents = eval_args.eval_num_agents
        
    if eval_args.eval_prandom is not None:
        all_args.prandom = eval_args.eval_prandom
    if eval_args.eval_urandom is not None:
        all_args.urandom = eval_args.eval_urandom
    if eval_args.eval_pfeeder is not None:
        all_args.pfeeder = eval_args.eval_pfeeder
    if eval_args.eval_run_name is not None:
        all_args.run_name = eval_args.eval_run_name

    if eval_args.eval_reset_patch_density is not None:
        all_args.reset_patch_density = eval_args.eval_reset_patch_density
    if eval_args.eval_step_patch_density is not None:
        all_args.step_patch_density = eval_args.eval_step_patch_density

    if eval_args.eval_eating_distribution_decay is not None:
        all_args.eating_distribution_decay_start = eval_args.eval_eating_distribution_decay
    if eval_args.eval_food_speed is not None:
        all_args.food_speed_start = eval_args.eval_food_speed
    if eval_args.eval_food_turn_std is not None:
        all_args.food_turn_std_start = eval_args.eval_food_turn_std
    if eval_args.eval_uniform_reset_food_density is not None:
        all_args.uniform_reset_food_density = eval_args.eval_uniform_reset_food_density
    if eval_args.eval_uniform_max_food_density is not None:
        all_args.uniform_max_food_density = eval_args.eval_uniform_max_food_density
    if eval_args.eval_uniform_step_food_density is not None:
        all_args.uniform_step_food_density = eval_args.eval_uniform_step_food_density

    if eval_args.eval_patchy_reset_food_density is not None:
        all_args.patchy_reset_food_density = eval_args.eval_patchy_reset_food_density
    if eval_args.eval_patchy_max_food_density is not None:
        all_args.patchy_max_food_density = eval_args.eval_patchy_max_food_density
    if eval_args.eval_patchy_step_food_density is not None:
        all_args.patchy_step_food_density = eval_args.eval_patchy_step_food_density

    if eval_args.eval_seed is not None:
        all_args.seed = eval_args.eval_seed

    # these are only for training
    all_args.food_speed_max = all_args.food_speed_start
    all_args.food_turn_std_max = all_args.food_turn_std_start
    all_args.food_speed_step = 0
    all_args.food_turn_std_step = 0

    # cfg overrides (these vars are set using cfg.py and require a different approach)
    if eval_args.eval_arena_size_max is not None:
        all_args.cfg_override["ENV_PARAMS"][
            "arena_size_max"
        ] = (eval_args.eval_arena_size_max, eval_args.eval_arena_size_max)
    if eval_args.eval_arena_size_min is not None:
        all_args.cfg_override["ENV_PARAMS"][
            "arena_size_min"
        ] = (eval_args.eval_arena_size_min, eval_args.eval_arena_size_min)
    if eval_args.eval_max_left_vergence is not None:
        all_args.cfg_override["FISH_CONSTANTS"][
            "max_left_vergence"
        ] = eval_args.eval_max_left_vergence
    if eval_args.eval_max_right_vergence is not None:
        all_args.cfg_override["FISH_CONSTANTS"][
            "max_right_vergence"
        ] = eval_args.eval_max_right_vergence
    if eval_args.eval_min_left_vergence is not None:
        all_args.cfg_override["FISH_CONSTANTS"][
            "min_left_vergence"
        ] = eval_args.eval_min_left_vergence
    if eval_args.eval_min_right_vergence is not None:
        all_args.cfg_override["FISH_CONSTANTS"][
            "min_right_vergence"
        ] = eval_args.eval_min_right_vergence

    if eval_args.eval_num_walkerbots is not None:
        all_args.num_walkerbots = eval_args.eval_num_walkerbots

    return all_args

def update_for_experiments(all_args, eval_args):
    """
    Update all_args based on additional experiments.
    """
    all_args.randomize_food_speed = eval_args.randomize_food_speed
    all_args.randomize_food_density = eval_args.randomize_food_density
    all_args.eval_limit_convergence = eval_args.eval_limit_convergence
    all_args.eval_limit_divergence = eval_args.eval_limit_divergence
    all_args.randomize_num_bots = eval_args.randomize_num_bots
    all_args.min_random = eval_args.min_random
    all_args.max_random = eval_args.max_random

def main(eval_args):
    """
    Run evaluation for a single configuration. (Run multiple times for multiple configurations.)

    Argument priority:
        1. eval_args passed into script (highest priority; overrides if not None)
        2. all_args from file
        3. env_args from file (used to override the current cfg file)
        4. values in current cfg.py file (might not match with cfg file used for training
           but helps with backwards compatibility if some attributes are added later)
    """
    run_dir = Path(eval_args.run_dir)
    log_dir = run_dir / "logs"

    all_args = read_args_from_file(log_dir)
    env_args = read_args_from_file(log_dir, "env_args", return_dict=True)
    old_cfg_args = get_old_cfg_args(env_args)
    all_args.cfg_override = old_cfg_args
    update_all_args(all_args, eval_args)
    if eval_args.additional_exps:
        update_for_experiments(all_args, eval_args)

    # all_args.seed = 1 # use fixed seed for eval for consistency across models

    envs = make_env(all_args, eval=True)
    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": None,
        "num_agents": all_args.num_agents,
        "device": setup_device(all_args),
        "run_dir": run_dir,
        "write_args_to_file": False,
        "additional_exps": eval_args.additional_exps,
        "additional_exps_name": eval_args.additional_exps_name,
    }

    Runner = SharedRunner if all_args.share_policy else SeparatedRunner
    if isinstance(Runner, SeparatedRunner):
        raise NotImplementedError

    runner = Runner(config)
    runner.render()
    
    run_name = runner.all_args.run_name
    log_exp_info(run_name, eval_args.run_dir, all_args, log_dir)

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "run_dir",
        type=str,
        help="Experiment directory, where logs/, models/ and outputs/ are subdirectories",
    )

    parser.add_argument("--eval_num_agents", type=int, default=None)

    parser.add_argument(
        "--eval_episode_length", type=int, default=50, help="Evaluation episode length"
    )
    parser.add_argument(
        "--save_vids",
        action="store_true",
        default=False,
        help="Whether to save GIFs for all eval episodes",
    )
    parser.add_argument(
        "--additional_exps",
        action="store_true",
        default=False,
        help="Run in additional exps mode, saving pkl files in a seperate marked folder",
    )
    parser.add_argument(
        "--additional_exps_name",
        type=str,
        default=None,
        help="Name for the additional exps folder. If None, a random string will be generated.",
    )
    parser.add_argument(
        "--n_rollout_threads", type=int, default=1, help="Number of rollout threads"
    )
    parser.add_argument(
        "--eval_render_episodes",
        type=int,
        default=1,
        help="Number of episodes to render during evaluation",
    )
    parser.add_argument(
        "--eval_seed",
        type=int_or_none,
        default=1,
        help="Seed for evaluation. If None, will use 1.",
    )
    parser.add_argument(
        "--eval_run_name",
        type=str,
        default=None,
        help="Name of the eval run, ideally a short unique identifier",
    )
    parser.add_argument("--num_vids_to_save", type=int, default=0)
    # Food related args
    parser.add_argument("--eval_pfeeder", type=float_or_none, default=None)
    parser.add_argument("--eval_prandom", type=float_or_none, default=None)
    parser.add_argument("--eval_urandom", type=float_or_none, default=None)
    parser.add_argument("--eval_food_preset_mode", type=str, default=None)
    parser.add_argument("--eval_eating_distribution_decay",
        type=float_or_none,
        default=None)
    parser.add_argument("--eval_reset_patch_density",
        type=float_or_none,
        default=None)
    parser.add_argument("--eval_step_patch_density",
        type=float_or_none,
        default=None)
    parser.add_argument("--eval_uniform_reset_food_density",
        type=float_or_none,
        default=None)
    parser.add_argument("--eval_uniform_max_food_density",
        type=float_or_none,
        default=None)
    parser.add_argument("--eval_uniform_step_food_density",
        type=float_or_none,
        default=None)
    parser.add_argument("--eval_patchy_reset_food_density",
        type=float_or_none,
        default=None)
    parser.add_argument("--eval_patchy_max_food_density",
        type=float_or_none,
        default=None)

    parser.add_argument("--eval_patchy_step_food_density", type=float_or_none, default=None)

    parser.add_argument("--eval_food_speed", type=float_or_none, default=cfg.ENV_PARAMS["food_speed"])
    parser.add_argument("--eval_food_turn_std", type=float_or_none, default=cfg.ENV_PARAMS["food_turn_std"])
    # Vergence related args
    parser.add_argument("--eval_max_left_vergence", type=float_or_none, default=cfg.FISH_CONSTANTS["max_left_vergence"])
    parser.add_argument("--eval_max_right_vergence", type=float_or_none, default=cfg.FISH_CONSTANTS["max_right_vergence"])
    parser.add_argument("--eval_min_left_vergence", type=float_or_none, default=cfg.FISH_CONSTANTS["min_left_vergence"])
    parser.add_argument("--eval_min_right_vergence", type=float_or_none, default=cfg.FISH_CONSTANTS["min_right_vergence"])
    # Arena + env related args
    parser.add_argument(
        "--eval_arena_size_max",
        type=int,
        default=None,
        help="Override ENV_PARAMS['arena_size_max'],"
        "pass as tuple, e.g. 70,70 or '(70,70)'",
    )
    parser.add_argument(
        "--eval_arena_size_min",
        type=int,
        default=None,
        help="Override ENV_PARAMS['arena_size_min'],"
        "pass as tuple, e.g. 70,70 or '(70,70)'",
    )
    parser.add_argument(
        "--eval_num_walkerbots",
        type=int,
        default=0,
        help="Number of walkerbots (non-fish agents) in the environment."
    )
    parser.add_argument(
        "--randomize_food_speed",
        action="store_true",
        default=False,
        help="Whether to randomize food speed in each episode",
    )
    parser.add_argument(
        "--randomize_food_density",
        action="store_true",
        default=False,
        help="Whether to randomize food density in each episode",
    )
    parser.add_argument(
        "--eval_limit_convergence",
        action="store_true",
        default=False,
        help="Whether to randomly limit convergence in the eval environment",
    )
    parser.add_argument(
        "--eval_limit_divergence",
        action="store_true",
        default=False,
        help="Whether to randomly limit divergence in the eval environment",
    )
    parser.add_argument(
        "--randomize_num_bots",
        action="store_true",
        default=False,
        help="Whether to randomize the number of bots in each episode",
    )
    parser.add_argument(
        "--min_random",
        type=float,
        default=0.0,
        help="Minimum value for randomization (if applicable)",
    )
    parser.add_argument(
        "--max_random",
        type=float,
        default=0.5,
        help="Maximum value for randomization (if applicable)",
    )

    parser.add_argument(
        "--eval_actor_load_path",
        type=str,
        default=None,
        help="Path to the actor model to load for evaluation. If None, will use the run directory. Does not pull from training step.",
    )

    args = parser.parse_args(args)
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)