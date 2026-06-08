# -------------------- Imports --------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from onpolicy.custom.forage.MAZFish import OBJECT_TYPES
from sklearn.linear_model import LinearRegression
import os

import cfg

def get_df_with_candidate_vars(dff):
    # Sort the DataFrame
    dff = dff.sort_values(
        by=["env_id", "episode_index", "agent_id", "time_step"]
    ).reset_index(drop=True)

    # Function to calculate nearest agent distances
    def calculate_nearest_agent_distances(dff):
        nearest_agent_distances = np.full(len(dff), np.inf)  # Initialize with infinity

        # Reset index to ensure it's consecutive for safe assignment
        dff = dff.reset_index(drop=True)

        grouped = dff.groupby(['env_id', 'episode_index', 'time_step'])

        for (env_id, episode_index, time_step), group in tqdm(grouped, desc="Calculating nearest agent distances"):
            positions = np.array(group['position'].tolist())
            indices = group.index.tolist()  # These are now aligned with the reset index
            pairwise_distances = np.linalg.norm(positions[:, np.newaxis, :] - positions[np.newaxis, :, :], axis=2)
            np.fill_diagonal(pairwise_distances, np.inf)
            nearest_distances = np.min(pairwise_distances, axis=1)

            nearest_agent_distances[indices] = nearest_distances

        return nearest_agent_distances

    # Calculate and add 'dist_to_nearest_agent' column
    dff['dist_to_nearest_agent'] = calculate_nearest_agent_distances(dff)

    # Re-sort the DataFrame
    dff = dff.sort_values(
        by=["env_id", "episode_index", "agent_id", "time_step"]
    ).reset_index(drop=True)

    print("After transformations:")
    print(dff.head())

    # Extract food positions for agent_id == 0
    food_positions_df = dff.loc[
        dff["agent_id"] == 0, ["episode_index", "env_id", "time_step", "food_ids", "food_positions"]
    ]
    dff.drop(columns=["food_ids", "food_positions"], inplace=True)
    print("Food positions (agent_id=0):")
    print(food_positions_df.head())

    # -------------------- Merging Food Positions --------------------
    # Merge food_positions_df with dff
    merged_df = pd.merge(
        dff, food_positions_df, on=["episode_index", "env_id", "time_step"], how="left"
    )

    print(
        "Food position matrix, unique shapes (before dropping broken rows):",
        pd.Series([np.array(fp).shape for fp in merged_df["food_positions"]]).unique(),
    )

    # Remove rows with no food
    rows_with_issue = merged_df[
        merged_df["food_positions"].apply(
            lambda x: np.array(x).shape == () or np.array(x).shape[0] == 0
        )
    ]
    print("NOTE: Num. rows with no food:", rows_with_issue.shape)

    # Drop rows with issues
    merged_df = merged_df.drop(rows_with_issue.index)

    print(
        "Food position matrix, unique shapes (after dropping broken rows):",
        pd.Series([np.array(fp).shape for fp in merged_df["food_positions"]]).unique(),
    )

    # -------------------- Calculating Distance and Angle to Closest Food --------------------
    # Function to calculate distance to closest food
    def distance_to_closest_food(row):
        agent_position = np.array(row["position"])
        food_positions = np.array(row["food_positions"])
        distances = np.linalg.norm(food_positions - agent_position, axis=1)
        return np.min(distances)

    # Function to calculate angle to closest food
    def angle_to_closest_food(row):
        agent_position = np.array(row["position"])
        agent_orientation = row["orientation"]
        food_positions = np.array(row["food_positions"])
        distances = np.linalg.norm(food_positions - agent_position, axis=1)
        closest_food_position = food_positions[np.argmin(distances)]
        vector_to_food = closest_food_position - agent_position
        angle = np.arctan2(vector_to_food[1], vector_to_food[0]) - agent_orientation
        return np.degrees(angle) % 360

    # Adding columns for distance and angle to closest food
    merged_df["distance_to_closest_food"] = merged_df.apply(
        distance_to_closest_food, axis=1
    )
    merged_df["angle_to_closest_food"] = merged_df.apply(angle_to_closest_food, axis=1)

    # Extract position coordinates
    merged_df["position_x"] = merged_df["position"].apply(lambda x: x[0])
    merged_df["position_y"] = merged_df["position"].apply(lambda x: x[1])

    # Placeholder for 'meeting_event' by shuffling 'eating_event'
    merged_df["meeting_event"] = merged_df["eating_event"].sample(frac=1).to_list()

    print("Columns after adding distance and angle to closest food:")
    print(merged_df.columns)
    print("Head of merged_df:")
    print(merged_df.head())

    
    return merged_df

def remove_episode_resets(dff):
    def count_food_positions(x):
        if isinstance(x, np.ndarray):
            return x.shape[0]
        else:
            return 0

    # Calculate 'num_food_items' for agent_id == 0
    num_food_items_df = dff[dff["agent_id"] == 0].copy()
    num_food_items_df["num_food_items"] = num_food_items_df["food_positions"].apply(count_food_positions)

    # Select relevant columns
    num_food_items_df = num_food_items_df[["episode_index", "env_id", "time_step", "num_food_items"]]

    # Merge 'num_food_items' back to the original dataframe
    dff = dff.merge(num_food_items_df, on=["episode_index", "env_id", "time_step"], how="left")
    dff["num_food_items"] = dff["num_food_items"].fillna(0)

    # Group the dataframe by 'episode_index' and 'env_id'
    grouped = dff.groupby(['episode_index', 'env_id'])

    def keep_longest_interval(group):
        # Get 'num_food_items' per time_step
        num_food_items = group.groupby('time_step')['num_food_items'].first().sort_index()

        # Identify time_steps when 'num_food_items' is zero
        zero_food_steps = num_food_items[num_food_items == 0].index.tolist()

        # Add start and end boundaries
        zero_food_steps = [-1] + zero_food_steps + [num_food_items.index.max() + 1]

        # Create intervals between resets
        intervals = []
        for i in range(len(zero_food_steps) - 1):
            start = zero_food_steps[i] + 1
            end = zero_food_steps[i+1] - 1
            if start <= end:
                intervals.append((start, end))

        # Find the longest interval
        if not intervals:
            return pd.DataFrame(columns=group.columns)
        longest_interval = max(intervals, key=lambda x: x[1] - x[0] + 1)
        start_step, end_step = longest_interval

        # Filter the group to keep only the longest interval
        filtered_group = group[(group['time_step'] >= start_step) & (group['time_step'] <= end_step)].copy()

        # Reset the 'time_step' index starting from zero
        filtered_group['time_step'] = filtered_group['time_step'] - start_step
        return filtered_group

    # Apply the function to each group and collect the results
    processed_groups = [keep_longest_interval(group) for _, group in grouped]

    # Concatenate all processed groups
    processed_df = pd.concat(processed_groups, ignore_index=True)

    # The resulting dataframe 'result_df' contains the longest intervals between resets for each episode, with 'time_step' indices reset starting from zero.

    return processed_df