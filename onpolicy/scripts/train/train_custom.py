#!/usr/bin/env python
import sys
import os
import socket
import setproctitle
import numpy as np
from pathlib import Path
import torch
from onpolicy.config import get_config
from onpolicy.envs.env_wrappers import SubprocVecEnv, DummyVecEnv

# Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0] + "/results")
from onpolicy.custom import MAForage



"""Train script for MAForage"""
def make_env(all_args, eval=False):
    def get_env_fn(rank):
        def init_env():
            env = MAForage(all_args)
            # Adjust seed calculation based on the eval flag
            if eval:
                env.seed(all_args.seed * 50000 + rank * 10000)
            else:
                env.seed(all_args.seed + rank * 1000)
            return env
        return init_env
    
    # Adjust thread count and vector environment based on the eval flag
    if eval:
        if all_args.n_eval_rollout_threads == 1:
            return DummyVecEnv([get_env_fn(0)])
        else:
            return SubprocVecEnv([get_env_fn(i) for i in range(all_args.n_eval_rollout_threads)])
    else:
        if all_args.n_rollout_threads == 1:
            return DummyVecEnv([get_env_fn(0)])
        else:
            return SubprocVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])


def parse_args(args, parser):
    parser.add_argument(
        "--scenario_name",
        type=str,
        default="simple",
    )
    parser.add_argument("--num_landmarks", type=int, default=3)
    parser.add_argument("--num_good_agents", type=int, default=2, help="number of good agents")
    parser.add_argument("--num_adversaries", type=int, default=2, help="number of adversaries")
    all_args = parser.parse_known_args(args)[0]
    print(all_args)
    return all_args


def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)

    all_args.env_name = "MAForage"
    all_args.algorithm_name = "rmappo"

    if all_args.algorithm_name == "rmappo":
        print("u are choosing to use rmappo, we set use_recurrent_policy to be True")
        all_args.use_recurrent_policy = True
        all_args.use_naive_recurrent_policy = False
    elif all_args.algorithm_name == "mappo":
        print(
            "u are choosing to use mappo, we set use_recurrent_policy & use_naive_recurrent_policy to be False"
        )
        all_args.use_recurrent_policy = False
        all_args.use_naive_recurrent_policy = False
    elif all_args.algorithm_name == "ippo":
        print("u are choosing to use ippo, we set use_centralized_V to be False")
        all_args.use_centralized_V = False
    else:
        raise NotImplementedError

    # cuda
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

    # run dir
    run_dir = (
        Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0] + "/results")
        / all_args.env_name
        / all_args.scenario_name
        / all_args.algorithm_name
        / all_args.experiment_name
    )
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    if not run_dir.exists():
        curr_run = "run1"
    else:
        exst_run_nums = [
            int(str(folder.name).split("run")[1])
            for folder in run_dir.iterdir()
            if str(folder.name).startswith("run")
        ]
        if len(exst_run_nums) == 0:
            curr_run = "run1"
        else:
            curr_run = "run%i" % (max(exst_run_nums) + 1)
    run_dir = run_dir / curr_run
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    setproctitle.setproctitle(
        str(all_args.algorithm_name)
        + "-"
        + str(all_args.env_name)
        + "-"
        + str(all_args.experiment_name)
        + "@"
        + str(all_args.user_name)
    )

    # seed
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    # env init
    envs = make_train_env(all_args)
    eval_envs = make_eval_env(all_args) if all_args.use_eval else None
    num_agents = all_args.num_agents

    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "num_agents": num_agents,
        "device": device,
        "run_dir": run_dir,
    }

    # run experiments
    if all_args.share_policy:
        print("Runner: Shared")
        # from onpolicy.runner.shared.mpe_runner import MPERunner as Runner
        from onpolicy.custom import CustomSharedRunner as Runner
    else:
        print("Runner: Separated")
        # from onpolicy.runner.separated.mpe_runner import MPERunner as Runner
        from onpolicy.custom import CustomSeparatedRunner as Runner

    runner = Runner(config)
    runner.run()

    print("Done training, now rendering....")
    runner.render()

    # post process
    envs.close()
    if all_args.use_eval and eval_envs is not envs:
        eval_envs.close()

    runner.writter.export_scalars_to_json(str(runner.log_dir + "/summary.json"))
    runner.writter.close()


if __name__ == "__main__":
    main(sys.argv[1:])
