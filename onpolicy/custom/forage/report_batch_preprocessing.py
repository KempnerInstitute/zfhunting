#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import sys
import argparse
import os
import re
import glob
import utils_report as ru
from scipy.special import expit as sigmoid
import utils_preprocess as up
import cfg
from pathlib import Path
from onpolicy.custom.forage.eval_ZFish import read_args_from_file

# Check if we're in interactive mode or batch mode
batchmode = False
if "ipykernel_launcher" in sys.argv[0]:
    print("Interactive mode")
else:
    batchmode = True
    print("Batch/CLI mode")

# outputs_folder = "/home/raaghav/zfish/onpolicy/custom/forage/results/rmappo-MultiAgentForagingEnv-check/20250624_164938_1_7103/outputs" 
outputs_folder = ru.get_latest_outputs_folder(base_path="./results/rmappo-MultiAgentForagingEnv-check/")

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

# Define the regular expression pattern to match "sr" followed by digits and then "_"
sr_pattern = r"sr(\d+)_"

# Search for the pattern in the folder name
match = re.search(sr_pattern, outputs_folder)

# Check if a match is found
if match:
    sensing_radius = int(match.group(1))  # Extract the matched digits
    print(f"The sensing radius is {sensing_radius}")
else:
    sensing_radius = 20  # Default value if no match is found
    print(f"No sensing radius found in the folder name, using default value of {sensing_radius}")

run_dir = Path(outputs_folder).parent
log_dir = run_dir / "logs"
print(f"Log directory: {log_dir}")
all_args = read_args_from_file(log_dir)

# Get all .pkl files in the folder excluding those with "flat" in the filename
pkl_files = [
    file for file in glob.glob(os.path.join(outputs_folder, "*.pkl"))
    if "flat" not in os.path.basename(file)
]

print(f"Found {len(pkl_files)} .pkl files to process.")

# Loop through each .pkl file
for pkl_file in pkl_files:
    print(f"Processing .pkl file: {pkl_file}")
    
    data = ru.load_data(pkl_file)
    print(f"Loaded .pkl file....")
    ru.print_column_shapes(data)

    # Preprocessing
    if "masks" in data.columns:
        data.drop(columns=["masks"], inplace=True)
    data["episode_index"] = pd.to_numeric(data["episode_index"])
    data["time_step"] = pd.to_numeric(data["time_step"])

    def get_num_envs(data):
        for col in ["actions", "observations", "rnn_states", "rewards", "infos"]:
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
        expanded_rows = []
        for idx, row in data.iterrows():
            for env_id in range(num_envs):
                new_row = row.copy()
                new_row["env_id"] = env_id
                for col in [
                    "actions",
                    "observations",
                    "rnn_states",
                    "rewards",
                    "infos",
                ]:
                    new_row[col] = row[col][env_id]
                expanded_rows.append(new_row)
        return pd.DataFrame(expanded_rows)

    def flatten_dataframe_agent(data):
        expanded_rows = []
        for idx, row in data.iterrows():
            for agent_idx in range(len(row["agent_id"])):
                new_row = row.copy()
                new_row["agent_id"] = row["agent_id"][agent_idx]
                for col in [
                    "actions",
                    "observations",
                    "rnn_states",
                    "rewards",
                    "infos",
                ]:
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
    for col in list_columns:
        dff[col] = cast_list_to_np_array(dff[col])

    # Un-list single-element lists in the rewards column
    dff["rewards"] = unlist_single_element_lists(dff["rewards"])

    print(all_args.binary_eye_state, all_args.discrete_actions, all_args.use_1dof_eyes)

    # Break out the actions column into separate columns
    if not all_args.binary_eye_state:
        if all_args.discrete_actions and not all_args.use_1dof_eyes:
            discrete_actions = dff['actions'].apply(lambda x: x[2])

            move_forward_turn_angle = discrete_actions.apply(lambda x: divmod(x, cfg.AGENT_PARAMS["num_turn_angles"]))

            # Assign the results to new columns
            dff['move_forward'] = move_forward_turn_angle.apply(lambda x: x[0])
            dff['turn_angle'] = move_forward_turn_angle.apply(lambda x: x[1])

            move_forward_vals = np.concatenate([[0], np.logspace(-1, 0, cfg.AGENT_PARAMS["num_speeds"])])
            turn_angle_vals = np.linspace(-1, 1, cfg.AGENT_PARAMS["num_turn_angles"])
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
        else:
            raise NotImplementedError(
                "Only discrete actions with 2D eyes, continuous actions with 2D eyes, or continuous actions with 1D eyes are supported."
            )
    else:
        if all_args.discrete_actions and not all_args.use_1dof_eyes:
            discrete_actions = dff['actions'].apply(lambda x: x[2])

            move_forward_turn_angle = discrete_actions.apply(lambda x: divmod(x, cfg.AGENT_PARAMS["num_turn_angles"]))

            # Assign the results to new columns
            dff['move_forward'] = move_forward_turn_angle.apply(lambda x: x[0])
            dff['turn_angle'] = move_forward_turn_angle.apply(lambda x: x[1])

            move_forward_vals = np.concatenate([[0], np.logspace(-1, 0, cfg.AGENT_PARAMS["num_speeds"])])
            turn_angle_vals = np.linspace(-1, 1, cfg.AGENT_PARAMS["num_turn_angles"])
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

    # Apply the function to each group
    dff['has_nearby'] = grouped.apply(lambda group: check_nearby_agents(group, sensing_radius)).explode().values

    # Save the flattened dataframe
    num_agents = len(dff['agent_id'].unique())
    flattened_pkl_file = pkl_file.replace(".pkl", f"_{num_agents}_flattened.pkl")
    merged_pkl_file = pkl_file.replace(".pkl", "_merged.pkl")

    dff.to_pickle(flattened_pkl_file)
    print(f"Saved flattened .pkl file: {flattened_pkl_file}")
    
    merged_df = dff # up.get_df_with_candidate_vars(dff)
    
    merged_df.to_pickle(merged_pkl_file)
    print(f"Saved merged .pkl file: {merged_pkl_file}")

print("Batch processing complete.")