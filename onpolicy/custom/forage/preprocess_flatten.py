"""
python -u preprocess_flatten.py [outputs_dir]
python -u preprocess_flatten.py [outputs_dir] --pkl_str [pkl_file_unique_identifier]

python -u preprocess_flatten.py ${OUTPUTS_DIR} 2>&1 | tee ${OUTPUTS_DIR}/preprocess_flatten.log
"""

import pandas as pd
import numpy as np
import os
import sys
import argparse
from pathlib import Path

import traceback

# from scipy.special import expit as sigmoid
from utils_general import sigmoid
import matplotlib.pyplot as plt

import utils_report as ru
from utils_report import (
    recurse_type_info,
    get_num_envs,
)
from utils_preprocess import get_df_with_candidate_vars
from onpolicy.custom.forage.eval_ZFish import read_args_from_file, get_old_cfg_args
from utils_behavior import analyze_vergence_during_food_tracking

from utils_general import (
    flatten_list_of_lists,
    cast_list_to_np_array,
    unlist_single_element_lists,
)
from utils_features import *
import traceback

import cfg

FISH_CONSTANTS = cfg.FISH_CONSTANTS
OBJECT_TYPES = cfg.OBJECT_TYPES
ENV_PARAMS = cfg.ENV_PARAMS
AGENT_PARAMS = cfg.AGENT_PARAMS
REWARDS = cfg.REWARDS
COLORS = cfg.COLORS


pkl_file = None
parser = argparse.ArgumentParser()
parser.add_argument(
    "run_dir",
    # nargs="?",  # requiring this argument for now
)
parser.add_argument(
    "--pkl_str",
    default=None,  # exclude the flattened pkl files
)
parser.add_argument(
    "--force",
    action="store_true",
    default=False,
)
parser.add_argument(
    "--delete_raw",
    action="store_true",
    default=False,
    help="Delete the raw pkl files after processing",
)
parser.add_argument(
    "--additional_exps",
    action="store_true",
    default=False,
    help="Run in additional_exps mode, which uses a different outputs folder structure",
)

args = parser.parse_args()
if args.additional_exps:
    outputs_folder = args.run_dir
else:
    outputs_folder = os.path.join(
        args.run_dir, "outputs"
    )  # Assumes outputs are in a folder named 'outputs' under the run_dir
print(f"Using outputs folder: {outputs_folder}")

# In case the cfg params or all_args are necessary for preprocessing
if args.additional_exps:
    run_dir = Path(outputs_folder).parent.parent.parent
else:
    run_dir = Path(outputs_folder).parent
log_dir = run_dir / "logs"
print(f"Using run folder: {run_dir}")
print(f"Using log folder: {log_dir}")

all_args = read_args_from_file(log_dir)
env_args = read_args_from_file(log_dir, "env_args", return_dict=True)
old_cfg_args = get_old_cfg_args(env_args)

print(old_cfg_args["AGENT_PARAMS"])

ENV_PARAMS.update(old_cfg_args["ENV_PARAMS"])
AGENT_PARAMS.update(old_cfg_args["AGENT_PARAMS"])
# REWARDS.update(old_cfg_args["REWARDS"])
OBJECT_TYPES.update(old_cfg_args["OBJECT_TYPES"])
if "FISH_CONSTANTS" in old_cfg_args:
    FISH_CONSTANTS.update(old_cfg_args["FISH_CONSTANTS"])

if args.pkl_str is not None:
    # pkl_file = ru.get_pkl_file_containing_str(outputs_folder, args.pkl_str)
    pkl_files = ru.get_all_raw_pkl_files_containing_str(outputs_folder, args.pkl_str)
else:
    pkl_files = ru.get_latest_raw_pkl_files(outputs_folder, args.force)

print(f"Using pkl file(s): {pkl_files}")

flattened_pkl_file = ru.get_expected_flattened_filename(pkl_files)  # OUTPUT FILENAME
print(f"Output will be written to flattened pkl file: {flattened_pkl_file}")

data = ru.get_df_from_pkls(pkl_files)
print(f"Loaded .pkl file(s)....")

ru.print_column_shapes(data)


# Preprocessing
if "masks" in data.columns:
    data.drop(columns=["masks"], inplace=True)
data["episode_index"] = pd.to_numeric(data["episode_index"])
data["time_step"] = pd.to_numeric(data["time_step"])

if "attn_mask" in data.columns and data["attn_mask"].apply(lambda x: x is None).all():
    data.drop(columns=["attn_mask"], inplace=True)
    print("Dropped attn_mask column, as all values are None")

def get_num_envs(data):
    cols = ["actions", "observations", "rnn_states", "rewards", "infos"]
    if "attn_mask" in data.columns:
        cols.append("attn_mask")
    for col in cols:
        first_element = data[col].iloc[0]
        if isinstance(first_element, np.ndarray):
            return first_element.shape[0]
        elif isinstance(first_element, list):
            return len(first_element)
    return 1  # Default to 1 if no valid columns are found

num_envs = get_num_envs(data)
print(f"Number of envs: {num_envs}")

def recurse_type_info(element):
    print(type(element))
    if isinstance(element, np.ndarray):
        print(element.shape)
    elif isinstance(element, list):
        print(len(element))
        if len(element) > 0:
            recurse_type_info(element[0])

for complex_column in data.columns:
    print()
    print(complex_column)
    first_element = data[complex_column].iloc[0]
    recurse_type_info(first_element)

def flatten_dataframe_env(data, num_envs=1):
    # Inspect NaN values in the original data
    print("=== NaN Analysis for Original Data ===")
    print(f"Total rows in data: {len(data)}")
    print(f"Columns with NaN values:")
    nan_counts = data.isnull().sum()
    nan_columns = nan_counts[nan_counts > 0]
    if len(nan_columns) > 0:
        for col, count in nan_columns.items():
            print(f"  {col}: {count} NaN values ({count/len(data)*100:.2f}%)")
    else:
        print("  No NaN values found in any column")

    # Check for empty/None values in complex columns
    print("\n=== Checking for None/empty values in complex columns ===")
    complex_cols = ["actions", "observations", "rnn_states", "rewards", "infos"]
    if "attn_mask" in data.columns:
        complex_cols.append("attn_mask")
    for col in complex_cols:
        if col in data.columns:
            none_count = data[col].isnull().sum()
            empty_count = data[col].apply(lambda x: x is None or (hasattr(x, '__len__') and len(x) == 0)).sum()
            print(f"{col}: {none_count} None values, {empty_count} empty values")
    expanded_rows = []
    for idx, row in data.iterrows():
        for env_id in range(num_envs):
            new_row = row.copy()
            new_row["env_id"] = env_id
            for col in complex_cols:
                new_row[col] = row[col][env_id]
            expanded_rows.append(new_row)
    return pd.DataFrame(expanded_rows)

def flatten_dataframe_agent(data):
    cols = ["actions", "observations", "rnn_states", "rewards", "infos"]
    if "attn_mask" in data.columns:
        cols.append("attn_mask")

    expanded_rows = []
    for idx, row in data.iterrows():
        for agent_idx in range(len(row["agent_id"])):
            new_row = row.copy()
            new_row["agent_id"] = row["agent_id"][agent_idx]
            for col in cols:
                new_row[col] = row[col][agent_idx]
            expanded_rows.append(new_row)
    return pd.DataFrame(expanded_rows)

# Flatten by env_id first
dff = flatten_dataframe_env(data, num_envs=num_envs)

# Flatten by agent_id next
dff = flatten_dataframe_agent(dff)

# Reset index
dff.reset_index(drop=True, inplace=True)

# Break out the infos column into separate columns
infos_df = pd.json_normalize(dff["infos"])
dff = dff.drop(columns=["infos"]).join(infos_df)

# Checking the types and structures after flattening
for complex_column in dff.columns:
    print(complex_column)
    first_element = dff[complex_column].iloc[0]
    recurse_type_info(first_element)

print(dff.columns)
print(dff.head())
print("dff.shape", dff.shape)

# Some more cleanup
def flatten_list_of_lists(column):
    """Flatten a column that is a list of lists"""
    return column.apply(
        lambda x: (
            [item for sublist in x for item in sublist] if isinstance(x, list) else x
        )
    )

def cast_list_to_np_array(column):
    """Cast a column that is a list to a numpy array"""
    return column.apply(lambda x: np.array(x) if isinstance(x, list) else x)

def unlist_single_element_lists(column):
    """Un-list columns that contain single-element lists"""
    return column.apply(lambda x: x[0] if isinstance(x, list) and len(x) == 1 else x)

# # Flatten list-of-lists columns
# list_of_lists_columns = [
#     "rnn_states",
# ]
# for col in list_of_lists_columns:
#     dff[col] = flatten_list_of_lists(dff[col])

# Cast list columns to numpy arrays
list_columns = [
    "actions",
    "observations",
    "rnn_states",
    "position",
    "food_positions",
]
if "attn_mask" in dff.columns:
    list_columns.append("attn_mask")

for col in list_columns:
    dff[col] = cast_list_to_np_array(dff[col])

# Un-list single-element lists in the rewards column
dff["rewards"] = unlist_single_element_lists(dff["rewards"])

# Break out the actions column into separate columns
if not all_args.binary_eye_state:
    if all_args.discrete_actions and not all_args.use_1dof_eyes:
        discrete_actions = dff['actions'].apply(lambda x: x[2])

        move_forward_turn_angle = discrete_actions.apply(lambda x: divmod(x, AGENT_PARAMS["num_turn_angles"]))

        # Assign the results to new columns
        dff['move_forward'] = move_forward_turn_angle.apply(lambda x: x[0])
        dff['turn_angle'] = move_forward_turn_angle.apply(lambda x: x[1])

        move_forward_vals = np.concatenate([[0], np.logspace(-1, 0, AGENT_PARAMS["num_speeds"])])
        turn_angle_vals = np.linspace(-1, 1, AGENT_PARAMS["num_turn_angles"])
        dff["move_forward"] = dff["move_forward"].apply(lambda x: move_forward_vals[int(x)])
        dff["turn_angle"] = dff["turn_angle"].apply(lambda x: turn_angle_vals[int(x)])
    elif not all_args.discrete_actions and not all_args.use_1dof_eyes:
        dff["move_forward"] = dff["actions"].apply(
            lambda x: x[2] if isinstance(x, np.ndarray) and len(x) >= 3 else None
        )
        dff["turn_angle"] = dff["actions"].apply(
            lambda x: x[3] if isinstance(x, np.ndarray) and len(x) >= 4 else None
        )
        # Apply transformations to 'move_forward' and 'turn_angle'
        dff["move_forward"] = dff["move_forward"].apply(lambda x: 1 / (1 + np.exp(-x)))
        dff["turn_angle"] = dff["turn_angle"].apply(lambda x: np.tanh(x))
    elif not all_args.discrete_actions and all_args.use_1dof_eyes:
        dff["move_forward"] = dff["actions"].apply(
            lambda x: x[1] if isinstance(x, np.ndarray) and len(x) >= 2 else None
        )
        dff["turn_angle"] = dff["actions"].apply(
            lambda x: x[2] if isinstance(x, np.ndarray) and len(x) >= 3 else None
        )
        # Apply transformations to 'move_forward' and 'turn_angle'
        dff["move_forward"] = dff["move_forward"].apply(lambda x: 1 / (1 + np.exp(-x)))
        dff["turn_angle"] = dff["turn_angle"].apply(lambda x: np.tanh(x))
    else:
        raise NotImplementedError(
            "Only discrete actions with 2D eyes, continuous actions with 2D eyes, or continuous actions with 1D eyes are supported."
        )
else:
    if all_args.discrete_actions and not all_args.use_1dof_eyes:
        discrete_actions = dff['actions'].apply(lambda x: x[2])

        move_forward_turn_angle = discrete_actions.apply(lambda x: divmod(x, AGENT_PARAMS["num_turn_angles"]))

        # Assign the results to new columns
        dff['move_forward'] = move_forward_turn_angle.apply(lambda x: x[0])
        dff['turn_angle'] = move_forward_turn_angle.apply(lambda x: x[1])

        move_forward_vals = np.concatenate([[0], np.logspace(-1, 0, AGENT_PARAMS["num_speeds"])])
        turn_angle_vals = np.linspace(-1, 1, AGENT_PARAMS["num_turn_angles"])
        dff["move_forward"] = dff["move_forward"].apply(lambda x: move_forward_vals[int(x)])
        dff["turn_angle"] = dff["turn_angle"].apply(lambda x: turn_angle_vals[int(x)])
    elif not all_args.discrete_actions:
        dff["move_forward"] = dff["actions"].apply(
            lambda x: x[0] if isinstance(x, np.ndarray) and len(x) >= 1 else None
        )
        dff["turn_angle"] = dff["actions"].apply(
            lambda x: x[1] if isinstance(x, np.ndarray) and len(x) >= 2 else None
        )
        # Apply transformations to 'move_forward' and 'turn_angle'
        dff["move_forward"] = dff["move_forward"].apply(lambda x: 1 / (1 + np.exp(-x)))
        dff["turn_angle"] = dff["turn_angle"].apply(lambda x: np.tanh(x))
    else:
        raise NotImplementedError(
            "Only discrete actions with 2D eyes, continuous actions with 2D eyes, or continuous actions with 1D eyes are supported."
        )

dff["turn_angle"] = dff["turn_angle"] * (1 - dff["move_forward"]) * (dff["move_forward"] > 0)

dff["eating_event"] = (
    dff.groupby(["env_id", "episode_index", "agent_id"])["energy"].diff() > 0
)
dff["displacement"] = (
    dff.groupby(["env_id", "episode_index", "agent_id"])["position"]
    .diff()
    .apply(lambda x: np.linalg.norm(x))
)

# Initialize the new column
dff['has_nearby'] = False

# Group by env_id, episode_index, and time_step for efficient processing
grouped = dff.groupby(['env_id', 'episode_index', 'time_step'])

def check_nearby_agents(group, radius):
    positions = np.stack(group['position'].values)
    dist_matrix = np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=2)
    np.fill_diagonal(dist_matrix, np.inf)  # ignore self-distance
    has_nearby = (dist_matrix < radius).any(axis=1)
    return has_nearby

def track_nearby_food_ids(dff, detection_radius=20):
    """
    Add a column 'nearby_food_ids' to the dataframe that contains a list of food IDs
    within a fixed distance from each agent at each time step.
    
    Parameters:
    - dff: DataFrame with agent data
    - detection_radius: Distance threshold for considering food as "nearby"
    
    Returns:
    - DataFrame with added 'nearby_food_ids' column
    """
    
    nearby_food_data = []
    
    for _, row in dff.iterrows():
        agent_pos = np.array(row['position'])
        food_positions = np.array(row['food_positions'])
        food_ids = row['food_ids']
        
        # Calculate distances to all food items
        if np.size(food_positions) and np.ndim(food_positions) > 0:
            distances = np.linalg.norm(food_positions - agent_pos, axis=1)
            
            # Find food within detection radius
            nearby_mask = distances <= detection_radius
            nearby_food_ids = [food_ids[i] for i in range(len(food_ids)) if nearby_mask[i]]
        else:
            nearby_food_ids = []
        
        nearby_food_data.append(nearby_food_ids)

    return nearby_food_data


#Fill in columns with missing agent data
try:
    dff = get_df_with_candidate_vars(dff)
except Exception as e:
    print("Exception in get_df_with_candidate_vars:", e)
    print(traceback.format_exc())


dff['nearby_food_ids'] = track_nearby_food_ids(dff, detection_radius=AGENT_PARAMS["food_detection_range"])

def get_binocular_food_ids(detected_food_ids):
    """
    Find food IDs that appear twice in the detected_food_ids (once for each eye).
    
    Parameters:
    - detected_food_ids: list of food IDs detected by both eyes
    
    Returns:
    - list of food IDs that appear exactly twice (binocular detection)
    """
    if not detected_food_ids:
        return []
    
    # Count occurrences of each food ID
    food_counts = {}
    for food_id in detected_food_ids:
        food_counts[food_id] = food_counts.get(food_id, 0) + 1
    
    # Return food IDs that appear exactly twice
    return [food_id for food_id, count in food_counts.items() if count == 2]

dff['binocular_food_ids'] = dff['detected_food_ids'].apply(get_binocular_food_ids)

dff["move_forward"] = dff["move_forward"] * cfg.FISH_CONSTANTS["max_speed"]
dff["turn_angle"] = dff["turn_angle"] * cfg.FISH_CONSTANTS["max_turn_speed"]

# Compute vergence angle and speed
perception_field = cfg.FISH_CONSTANTS["perception_field"]
dff['vergence_angle'] = dff['left_eye_angle'] - dff['right_eye_angle'] + perception_field
dff['speed'] = dff['move_forward']
dff['vergence_angle_deg'] = dff['vergence_angle'] * 180 / np.pi
# Normalize orientation to be between -pi and pi
dff['orientation'] = np.arctan2(np.sin(dff['orientation']), np.cos(dff['orientation']))

dff = dff.sort_values(
    by=["env_id", "episode_index", "agent_id", "time_step"]
).reset_index(drop=True)

feature_names = []
if not args.additional_exps:
    # Add additional features
    tracking_sequences_df = analyze_vergence_during_food_tracking(dff)

    
    dff, feature_names = add_hunting(dff, feature_names, tracking_sequences_df)
    print(dff["food_positions"])
    dff, feature_names = add_distance_angle_to_food_hunting(dff, feature_names, tracking_sequences_df)
    dff, feature_names = add_distance_to_wall_circular(dff, feature_names)
    dff, feature_names = add_distance_angle_to_closest_walker(dff, feature_names)
    dff, feature_names = add_walkerbot_visible(dff, feature_names)

dff, feature_names = add_group_statistics(dff, feature_names)

# Apply the function to each group
# dff['has_nearby'] = grouped.apply(lambda group: check_nearby_agents(group, sensing_radius)).explode().values

print("Saving flattened .pkl file ...")
dff.to_pickle(flattened_pkl_file)
print(f"Saved flattened .pkl file: {flattened_pkl_file}")

# Optionally delete the raw pkl files
if args.delete_raw:
    print("Deleting corresponding raw pkl files...")
    for pkl_path_str in pkl_files:
        try:
            os.remove(pkl_path_str)
        except FileNotFoundError:
            pass
    print("Deleted corresponding raw pkl files.")