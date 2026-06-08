# Updates performance metrics after a run has already completed training and performance_metrics.txt exists
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from matplotlib.animation import FuncAnimation
import numpy as np
import sys
import argparse
import os
import glob
import utils_report as ru
import utils_behavior as ub
from utils_behavior import calculate_polarization, calculate_cohesion, theil_index
import cfg

# Check if we're in interactive mode or batch mode
batchmode = False
if "ipykernel_launcher" in sys.argv[0]:
    print("Interactive mode")
else:
    batchmode = True
    print("Batch/CLI mode")
    plt.ioff()
    # Parses the command line arguments below


def get_latest_flat_pkl_file(input_dir="./"):
    pkl_files = glob.glob(input_dir + "/*.pkl")
    pkl_files = [f for f in pkl_files if "flat" in f]
    if not pkl_files:
        raise FileNotFoundError("No .pkl files found in the current directory.")
    latest_pkl_file = max(pkl_files, key=os.path.getctime)
    return latest_pkl_file

default_dir = "./results/rmappo-MultiAgentForagingEnv-check/"

outputs_folder = ru.get_latest_outputs_folder(default_dir)
    # "/srv/marl/satsingh/marl_fish/20241106_182038/outputs/"  # with biting
    # "/home/satsingh/srv/marl/satsingh/marl_fish/rmappo-MultiAgentFishEnv-114/outputs/"  # GOOD
    # "/home/satsingh/srv/marl/satsingh/marl_fish/20241013_202859/outputs"  # same as rmappo-MultiAgentFishEnv-114
    #     "/home/satsingh/srv/marl/satsingh/marl_fish/20241013_202859/outputs/" # BEST
    #     "/home/satsingh/srv/marl/satsingh/marl_fish/20241016_202055/outputs/" # OK
    #     "/home/satsingh/srv/marl/satsingh/marl_fish/20241016_202056/outputs/" # OK
    #     "/home/satsingh/srv/marl/satsingh/marl_fish/20241016_202053/outputs/" # OK

if batchmode:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "outputs_folder",
        default=outputs_folder,
        nargs="?",
    )
    args = parser.parse_args()
    outputs_folder = args.outputs_folder

print(f"Using outputs folder: {outputs_folder}")

flat_pkl_file = get_latest_flat_pkl_file(outputs_folder)
print(f"Using .pkl file: {flat_pkl_file}")

dff = pd.read_pickle(flat_pkl_file)
# ru.print_column_shapes(dff)
print("dff.shape", dff.shape)
print("dff.columns", dff.columns)

dff.drop(columns=["rnn_states"], inplace=True)


# In[3]:


dff = dff.sort_values(
    by=["env_id", "episode_index", "agent_id", "time_step"]
).reset_index(drop=True)

import os

# Create a figure results folder
results_folder = f"{outputs_folder}/figures"
os.makedirs(results_folder, exist_ok=True)

print(f"Created folder: {results_folder}")


# In[7]:


# Speed is calculated via displacement so has some bugs -- more interesting to look at move_forward

dff['speed'] = dff['move_forward']

# Calculate move_forward variance only where food is detected
food_detected_mask = dff['detected_food_ids'].apply(lambda x: len(x) > 0 if isinstance(x, list) else False)
if food_detected_mask.any():
    move_forward_variance_with_food = dff.loc[food_detected_mask, 'move_forward'].var()
else:
    move_forward_variance_with_food = 0.0
    print("No food detected instances found in the data")
print(f"Move forward variance when food is detected: {move_forward_variance_with_food:.4f}")

# Calculate move_forward variance only where no food is detected
no_food_detected_mask = ~food_detected_mask
if no_food_detected_mask.any():
    move_forward_variance_with_no_food = dff.loc[no_food_detected_mask, 'move_forward'].var()
else:
    move_forward_variance_with_no_food = 0.0
    print("No instances without food detected found in the data")
print(f"Move forward variance when no food is detected: {move_forward_variance_with_no_food:.4f}")

# Check if the metric already exists in the file
performance_file = os.path.join(results_folder, "performance_metrics.txt")
metric_exists = False
if os.path.exists(performance_file):
    with open(performance_file, "r") as f:
        content = f.read()
        if "Move forward variance when food detected" in content:
            metric_exists = True

if not metric_exists:
    with open(performance_file, "a") as f:
        f.write(f"Move forward variance when food detected: {move_forward_variance_with_food:.4f}\n")

metric_exists = False
if os.path.exists(performance_file):
    with open(performance_file, "r") as f:
        content = f.read()
        if "Move forward variance when no food detected" in content:
            metric_exists = True

if not metric_exists:
    with open(performance_file, "a") as f:
        f.write(f"Move forward variance when no food detected: {move_forward_variance_with_no_food:.4f}\n")


# Calculate move_forward variance only when hunting
hunting_mask = dff['hunting'] == True
if hunting_mask.any():
    move_forward_variance_while_hunting = dff.loc[hunting_mask, 'move_forward'].var()
else:
    move_forward_variance_while_hunting = 0.0
    print("No hunting instances found in the data")
print(f"Move forward variance while hunting: {move_forward_variance_while_hunting:.4f}")


metric_exists = False
if os.path.exists(performance_file):
    with open(performance_file, "r") as f:
        content = f.read()
        if "Move forward variance while hunting" in content:
            metric_exists = True

if not metric_exists:
    with open(os.path.join(results_folder, "performance_metrics.txt"), "a") as f:
        f.write(f"Move forward variance while hunting: {move_forward_variance_while_hunting:.4f}\n")

# Calculate move_forward variance only when not hunting
not_hunting_mask = dff['hunting'] == False
if not_hunting_mask.any():
    move_forward_variance_while_not_hunting = dff.loc[not_hunting_mask, 'move_forward'].var()
    move_forward_speed_while_not_hunting = dff.loc[not_hunting_mask, 'move_forward'].mean()
else:
    move_forward_variance_while_not_hunting = 0.0
    print("No non-hunting instances found in the data")
print(f"Move forward variance while not hunting: {move_forward_variance_while_not_hunting:.4f}")


metric_exists = False
if os.path.exists(performance_file):
    with open(performance_file, "r") as f:
        content = f.read()
        if "Move forward variance while not hunting" in content:
            metric_exists = True

if not metric_exists:
    with open(os.path.join(results_folder, "performance_metrics.txt"), "a") as f:
        f.write(f"Move forward variance while not hunting: {move_forward_variance_while_not_hunting:.4f}\n")

metric_exists = False
if os.path.exists(performance_file):
    with open(performance_file, "r") as f:
        content = f.read()
        if "Move forward speed while not hunting" in content:
            metric_exists = True

if not metric_exists:
    with open(os.path.join(results_folder, "performance_metrics.txt"), "a") as f:
        f.write(f"Move forward speed while not hunting: {move_forward_speed_while_not_hunting:.4f}\n")

# Get eye turn action
dff["eye_turn"] = dff["actions"].apply(
            lambda x: x[0] if isinstance(x, np.ndarray) and len(x) >= 1 else None
        )
dff["eye_turn"] = dff["eye_turn"].apply(lambda x: np.tanh(x))
dff["eye_turn"] = dff["eye_turn"].apply(lambda x: x * cfg.FISH_CONSTANTS["max_eye_turn_speed"])

dff["turn_angle_abs"] = dff["turn_angle"].abs()
dff['turn_change'] = dff.groupby(['env_id', 'episode_index', 'agent_id'])['turn_angle'].diff()
dff['abs_turn_change'] = dff['turn_change'].abs()
dff["eye_turn_abs"] = dff["eye_turn"].abs()


metric_exists = False
if os.path.exists(performance_file):
    with open(performance_file, "r") as f:
        content = f.read()
        if "Abs eye turn mean" in content:
            metric_exists = True

if not metric_exists:
    with open(os.path.join(results_folder, "performance_metrics.txt"), "a") as f:
        f.write(f"Abs eye turn mean: {dff['eye_turn_abs'].mean():.4f}\n")
        f.write(f"Abs eye turn var: {dff['eye_turn_abs'].var():.4f}\n")
        f.write(f"Abs eye turn min: {dff['eye_turn_abs'].min():.4f}\n")
        f.write(f"Abs eye turn max: {dff['eye_turn_abs'].max():.4f}\n")

metric_exists = False
if os.path.exists(performance_file):
    with open(performance_file, "r") as f:
        content = f.read()
        if "Abs turn angle mean" in content:
            metric_exists = True

if not metric_exists:
    with open(os.path.join(results_folder, "performance_metrics.txt"), "a") as f:
        f.write(f"Abs turn angle mean: {dff['turn_angle_abs'].mean():.4f}\n")
        f.write(f"Abs turn angle var: {dff['turn_angle_abs'].var():.4f}\n")
        f.write(f"Abs turn angle min: {dff['turn_angle_abs'].min():.4f}\n")
        f.write(f"Abs turn angle max: {dff['turn_angle_abs'].max():.4f}\n")

# Calculate eye vergence change within each episode
dff['vergence_change'] = dff.groupby(['env_id', 'episode_index', 'agent_id'])['vergence_angle'].diff()

# Fill NaN values (first timestep of each episode) with 0
dff['vergence_change'] = dff['vergence_change'].fillna(0)

dff['abs_vergence_change'] = dff['vergence_change'].abs()

metric_exists = False
if os.path.exists(performance_file):
    with open(performance_file, "r") as f:
        content = f.read()
        if "Abs vergence change mean" in content:
            metric_exists = True

if not metric_exists:
    with open(os.path.join(results_folder, "performance_metrics.txt"), "a") as f:
        f.write(f"Abs vergence change mean: {dff['abs_vergence_change'].mean():.4f}\n")
        f.write(f"Abs vergence change var: {dff['abs_vergence_change'].var():.4f}\n")
        f.write(f"Abs vergence change min: {dff['abs_vergence_change'].min():.4f}\n")
        f.write(f"Abs vergence change max: {dff['abs_vergence_change'].max():.4f}\n")
        f.write(f"Abs vergence change median: {dff['abs_vergence_change'].median():.4f}\n")
        f.write(f"Abs vergence change 95th percentile: {dff['abs_vergence_change'].quantile(0.95):.4f}\n")
        f.write(f"Abs vergence change 5th percentile: {dff['abs_vergence_change'].quantile(0.05):.4f}\n")


metric_exists = False
if os.path.exists(performance_file):
    with open(performance_file, "r") as f:
        content = f.read()
        if "Abs turn change mean" in content:
            metric_exists = True

if not metric_exists:
    with open(os.path.join(results_folder, "performance_metrics.txt"), "a") as f:
        f.write(f"Abs turn change mean: {dff['abs_turn_change'].mean():.4f}\n")
        f.write(f"Abs turn change var: {dff['abs_turn_change'].var():.4f}\n")
        f.write(f"Abs turn change min: {dff['abs_turn_change'].min():.4f}\n")
        f.write(f"Abs turn change max: {dff['abs_turn_change'].max():.4f}\n")
        f.write(f"Abs turn change median: {dff['abs_turn_change'].median():.4f}\n")
        f.write(f"Abs turn change 95th percentile: {dff['abs_turn_change'].quantile(0.95):.4f}\n")
        f.write(f"Abs turn change 5th percentile: {dff['abs_turn_change'].quantile(0.05):.4f}\n")
