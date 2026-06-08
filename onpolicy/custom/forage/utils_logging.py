import os
import csv
from datetime import datetime


def log_on_done(env):
    # Grab the folder where Runner is writing its logs:
    log_dir = env.all_args["run_dir"] + "/logs"
    os.makedirs(log_dir, exist_ok=True)
    path = os.path.join(log_dir, "env_params.csv")
    write_header = not os.path.exists(path)

    # Collect exactly the fields you care about
    row = {
        "timestamp": datetime.now().isoformat(),
        "arena_type": env.arena.__class__.__name__,
        "food_scaling_factor": round(env.food_scaling_factor, 3),
        "arena_size_x": env.arena_size[0],
        "arena_size_y": env.arena_size[1],
        "eating_distribution_decay": round(env.eating_distribution_decay, 3),
        "food_speed": round(env.food_speed, 3),
        "food_turn_std": round(env.food_turn_std, 3),
        "current_episode": env.current_episode,
        "cumulative_reward": round(env.agent_objects[0].cumulative_reward, 3),  # only agent 0 for single-agent experiments
    }

    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=row.keys())
        if write_header:
            w.writeheader()
        w.writerow(row)