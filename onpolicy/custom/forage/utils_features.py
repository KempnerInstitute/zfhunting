# These utilities typically operate on the flattened dff

import numpy as np
import pandas as pd
import tqdm
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from onpolicy.custom.forage import cfg

# def add_group_statistics(df, feature_names):
#     """
#     Add nearest-neighbor swimming statistics in place for each agent.

#     Returns the updated DataFrame and new feature_names list.
#     """
#      # Preallocate columns (filled with NaN)
#     df["closest_agent_id"] = np.nan
#     df["nn_distance"] = np.nan
#     df["nn_cosine_orientation"] = np.nan
#     df["nn_orientation_diff"] = np.nan
#     df["nn_heading_world"] = np.nan
#     df["nn_heading_relative"] = np.nan

#     # Process each (env_id, episode_index, time_step) group
#     for (env_id, episode_index, time_step), idx in df.groupby(
#         ["env_id", "episode_index", "time_step"]
#     ).groups.items():

#         group = df.loc[idx]
#         positions = np.vstack(group["position"].to_numpy())   # shape (N, 2)
#         orientations = group["orientation"].to_numpy()        # shape (N,)
#         agent_ids = group["agent_id"].to_numpy()

#         # --- Pairwise distances ---
#         diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
#         dists = np.linalg.norm(diff, axis=-1)
#         np.fill_diagonal(dists, np.inf)

#         # --- Nearest neighbor ---
#         closest_indices = np.argmin(dists, axis=1)
#         closest_agents = agent_ids[closest_indices]
#         closest_distances = dists[np.arange(len(agent_ids)), closest_indices]

#         # --- Orientation difference ---
#         closest_orientations = orientations[closest_indices]
#         orientation_diffs = (orientations - closest_orientations + np.pi) % (2 * np.pi) - np.pi
#         nn_cosine_orientation = np.cos(orientation_diffs)

#         # --- Heading to nearest neighbor ---
#         nearest_diff = positions[closest_indices] - positions
#         nn_heading_world = np.arctan2(nearest_diff[:, 1], nearest_diff[:, 0])  # absolute angle
#         nn_heading_relative = (nn_heading_world - orientations + np.pi) % (2 * np.pi) - np.pi

#         # --- Assign values directly into df ---
#         df.loc[idx, "closest_agent_id"] = closest_agents
#         df.loc[idx, "nn_distance"] = closest_distances
#         df.loc[idx, "nn_cosine_orientation"] = nn_cosine_orientation
#         df.loc[idx, "nn_orientation_diff"] = orientation_diffs
#         df.loc[idx, "nn_heading_world"] = nn_heading_world
#         df.loc[idx, "nn_heading_relative"] = nn_heading_relative

#     # Add new feature names
#     feature_names += [
#         "nn_distance",
#         "nn_cosine_orientation",
#         "nn_orientation_diff",
#         "nn_heading_world",
#         "nn_heading_relative",
#     ]

#     return df, feature_names

#Faster version: if there are errors, try switching back to the above version
def add_group_statistics(df, feature_names):
    """
    Fast nearest-neighbor computation using NumPy grouping.
    Assumes it's OK to sort df by (env_id, episode_index, time_step).
    """

    # ---------- Sort df once ----------
    df = df.sort_values(["env_id", "episode_index", "time_step"], kind="mergesort")
    df = df.reset_index(drop=True)

    # ---------- Extract columns into NumPy ----------
    positions = np.stack(df["position"].to_numpy())   # (N,2)
    orientations = df["orientation"].to_numpy()       # (N,)
    agent_ids = df["agent_id"].to_numpy()             # (N,)
    keys = df[["env_id", "episode_index", "time_step"]].to_numpy()

    N = len(df)

    # ---------- Allocate outputs ----------
    closest_agent_id      = np.full(N, np.nan)
    nn_distance           = np.full(N, np.nan)
    nn_cosine_orientation = np.full(N, np.nan)
    nn_orientation_diff   = np.full(N, np.nan)
    nn_heading_world      = np.full(N, np.nan)
    nn_heading_relative   = np.full(N, np.nan)

    # ---------- Find group boundaries ----------
    change = np.any(keys[1:] != keys[:-1], axis=1)
    bounds = np.flatnonzero(change) + 1
    bounds = np.r_[0, bounds, N]

    # ---------- Process groups ----------
    for start, end in zip(bounds[:-1], bounds[1:]):
        size = end - start
        if size <= 1:
            continue

        sl = slice(start, end)
        pos = positions[sl]     # (G,2)
        ori = orientations[sl]  # (G,)
        aid = agent_ids[sl]     # (G,)

        # -- Pairwise distances --
        diff = pos[:, None, :] - pos[None, :, :]
        dists = np.linalg.norm(diff, axis=-1)
        np.fill_diagonal(dists, np.inf)

        closest_idx = np.argmin(dists, axis=1)
        closest_agents = aid[closest_idx]
        closest_dists = dists[np.arange(size), closest_idx]

        # -- Orientation difference --
        nearest_ori = ori[closest_idx]
        dtheta = (ori - nearest_ori + np.pi) % (2*np.pi) - np.pi
        cos_rel = np.cos(dtheta)

        # -- Headings --
        dif = pos[closest_idx] - pos
        heading_world = np.arctan2(dif[:, 1], dif[:, 0])
        heading_rel   = (heading_world - ori + np.pi) % (2*np.pi) - np.pi

        # -- Write results --
        closest_agent_id[sl]      = closest_agents
        nn_distance[sl]           = closest_dists
        nn_cosine_orientation[sl] = cos_rel
        nn_orientation_diff[sl]   = dtheta
        nn_heading_world[sl]      = heading_world
        nn_heading_relative[sl]   = heading_rel

    # ---------- Store back into df ----------
    df["closest_agent_id"] = closest_agent_id
    df["nn_distance"] = nn_distance
    df["nn_cosine_orientation"] = nn_cosine_orientation
    df["nn_orientation_diff"] = nn_orientation_diff
    df["nn_heading_world"] = nn_heading_world
    df["nn_heading_relative"] = nn_heading_relative

    feature_names += [
        "nn_distance",
        "nn_cosine_orientation",
        "nn_orientation_diff",
        "nn_heading_world",
        "nn_heading_relative",
    ]

    return df, feature_names

def add_distance_to_wall_rectangular(df, feature_names):
    # Make sure that arena_size is available on each row
    df = df.sort_values(by=["env_id", "episode_index", "time_step"])
    df["arena_size"] = df.groupby(["env_id", "episode_index"])["arena_size"].ffill()

    if {"position_x", "position_y", "arena_size"}.issubset(df.columns):
        try:
            df["distance_to_wall"] = df.apply(
                lambda row: (
                    min(
                        row["position_x"],
                        row["arena_size"][0] - row["position_x"],
                        row["position_y"],
                        row["arena_size"][1] - row["position_y"],
                    )
                    if not pd.isna(row["arena_size"])
                    else np.nan
                ),
                axis=1,
            )
            feature_names.append("distance_to_wall")
        except Exception as e:
            print(f"Could not calculate distance to wall: {e}")
            print(
                "Ensure 'position_x', 'position_y', and 'arena_size' are present in the DataFrame."
            )
    return df, feature_names


def add_distance_angle_to_closest_food(df, feature_names, add_binned_features=True):
    if {"distance_to_closest_food", "angle_to_closest_food"}.issubset(df.columns):
        feature_names.extend(["distance_to_closest_food", "angle_to_closest_food"])

        # Convert degrees to radians if needed for food angle
        if df["angle_to_closest_food"].abs().max() > 2 * np.pi:
            # If angle is in degrees, convert to radians
            df["angle_to_closest_food_rad"] = np.radians(df["angle_to_closest_food"])
        else:
            # If angle is already in radians, just copy
            df["angle_to_closest_food_rad"] = df["angle_to_closest_food"]

        # Add circular representation (cos and sin)
        df["angle_to_closest_food_cos"] = np.cos(df["angle_to_closest_food_rad"])
        df["angle_to_closest_food_sin"] = np.sin(df["angle_to_closest_food_rad"])
        # Add a virtual feature name for the circular representation
        feature_names.append("angle_to_closest_food_circular")

        if add_binned_features:
            df["distance_to_food_binned"] = pd.cut(
                df["distance_to_closest_food"],
                bins=[0, 10, 15, 40, 100, np.inf],
                labels=range(1, 6),
            ).astype(int)
            feature_names.append("distance_to_food_binned")
            df["angle_to_food_binned"] = pd.cut(
                df["angle_to_closest_food"],
                bins=np.linspace(-180, 180, 9),
                labels=range(1, 9),
                include_lowest=True,
            ).astype(int)
            feature_names.append("angle_to_food_binned")
    return df, feature_names


def add_velocity_to_nearest_agent(df, feature_names):
    df["approach_nearest_velocity"] = (
        df.groupby(["episode_index", "env_id", "agent_id"])["distance_to_nearest_agent"]
        .diff()
        .fillna(0)
    )
    feature_names.append("approach_nearest_velocity")
    return df, feature_names

def add_hunting(df, feature_names, tracking_sequences_df):
    df["hunting"] = False

    # Iterate through tracking_results to mark hunting time steps
    for _, row in tracking_sequences_df.iterrows():
        mask = (
            (df["env_id"] == row["env_id"]) &
            (df["episode_index"] == row["episode_index"]) &
            (df["agent_id"] == row["agent_id"]) &
            (df["time_step"] >= row["start_time_step"]) &
            (df["time_step"] < row["start_time_step"] + row["tracking_duration"])
        )
        df.loc[mask, "hunting"] = True

    feature_names.append("hunting")
    return df, feature_names

def add_distance_angle_to_food_hunting(df, feature_names, tracking_sequences_df):
    # Filter successful hunts
    successful_hunts = tracking_sequences_df[tracking_sequences_df["outcome"] == "success"]

    # Iterate through successful hunts
    for _, hunt in successful_hunts.iterrows():
        # Filter the corresponding time steps in dff
        mask = (
            (df["env_id"] == hunt["env_id"]) &
            (df["episode_index"] == hunt["episode_index"]) &
            (df["agent_id"] == hunt["agent_id"]) &
            (df["time_step"] >= hunt["start_time_step"]) &
            (df["time_step"] < hunt["start_time_step"] + hunt["tracking_duration"])
        )
        hunt_steps = df[mask]

        # Get the food position
        food_id = hunt["food_id"]

        # Calculate distance and orientation for each time step
        for t, row in hunt_steps.iterrows():
            idx = t - hunt_steps.index[0]  # Get the relative index for the current time step
            agent_position = row["position"]
            agent_orientation = row["orientation"]
            agent_orientation = (agent_orientation + np.pi) % (2 * np.pi) - np.pi  # Normalize to [-pi, pi]

            food_positions = hunt_steps["food_positions"].iloc[idx]
            food_ids = hunt_steps["food_ids"].iloc[idx]
            food_position = food_positions[np.where(np.array(food_ids) == food_id)[0][0]]

            # Calculate distance
            distance = np.linalg.norm(np.array(agent_position) - np.array(food_position))
            # Find the row index in the dataframe that corresponds to this hunt step
            df_mask = (
                (df["env_id"] == hunt["env_id"]) &
                (df["episode_index"] == hunt["episode_index"]) &
                (df["agent_id"] == hunt["agent_id"]) &
                (df["time_step"] == row["time_step"])
            )
            df.loc[df_mask, "distance_to_food_hunting"] = distance

            # Calculate orientation
            vector_to_food = np.array(food_position) - np.array(agent_position)
            orientation_to_food = np.arctan2(vector_to_food[1], vector_to_food[0]) - agent_orientation
            orientation_to_food = (orientation_to_food + np.pi) % (2 * np.pi) - np.pi  # Normalize to [-pi, pi]
            df.loc[df_mask, "angle_to_food_hunting_deg"] = np.rad2deg(orientation_to_food)

    feature_names.append("distance_to_food_hunting")
    feature_names.append("angle_to_food_hunting_deg")

    return df, feature_names

def add_walkerbot_visible(df, feature_names):
    if "walkerbot_positions" in df.columns:
        df["walkerbot_visible"] = False
        
        for idx, row in df.iterrows():
            agent_position = np.array(row["position"])
            agent_orientation = row["orientation"]
            left_eye_angle = row["left_eye_angle"]
            right_eye_angle = row["right_eye_angle"]
            walkerbot_positions = row["walkerbot_positions"]

            if walkerbot_positions is None:
                continue

            walkerbot_positions = np.array(walkerbot_positions)
            if not pd.isna(walkerbot_positions).any() and walkerbot_positions.size > 0:
                for walkerbot_pos in walkerbot_positions:
                    walkerbot_pos = np.array(walkerbot_pos)
                    
                    # Calculate vector from agent to walkerbot
                    vector_to_walkerbot = walkerbot_pos - agent_position
                    angle_to_walkerbot = np.arctan2(vector_to_walkerbot[1], vector_to_walkerbot[0])
                    
                    # Calculate relative angle to agent orientation
                    relative_angle = angle_to_walkerbot - agent_orientation
                    relative_angle = (relative_angle + np.pi) % (2 * np.pi) - np.pi  # Normalize to [-pi, pi]
                    
                    # Forward direction
                    forward_x = np.cos(agent_orientation) * cfg.FISH_CONSTANTS["eye_forward_offset"]
                    forward_y = np.sin(agent_orientation) * cfg.FISH_CONSTANTS["eye_forward_offset"]

                    # Side direction (perpendicular to forward)
                    side_x = -np.sin(agent_orientation) * cfg.FISH_CONSTANTS["eye_separation"] / 2
                    side_y = np.cos(agent_orientation) * cfg.FISH_CONSTANTS["eye_separation"] / 2

                    # Left eye position (to the left when looking forward)
                    left_eye_position = agent_position + np.array(
                        [forward_x - side_x, forward_y - side_y]
                    )

                    # Right eye position (to the right when looking forward)
                    right_eye_position = agent_position + np.array(
                        [forward_x + side_x, forward_y + side_y]
                    )
                    
                    # Check visibility from left eye
                    vector_from_left_eye = walkerbot_pos - left_eye_position
                    angle_from_left_eye = np.arctan2(vector_from_left_eye[1], vector_from_left_eye[0])
                    relative_angle_left = angle_from_left_eye - (agent_orientation + left_eye_angle)
                    relative_angle_left = (relative_angle_left + np.pi) % (2 * np.pi) - np.pi
                    
                    # Check visibility from right eye
                    vector_from_right_eye = walkerbot_pos - right_eye_position
                    angle_from_right_eye = np.arctan2(vector_from_right_eye[1], vector_from_right_eye[0])
                    relative_angle_right = angle_from_right_eye - (agent_orientation + right_eye_angle)
                    relative_angle_right = (relative_angle_right + np.pi) % (2 * np.pi) - np.pi
                    
                    # Check if walkerbot is visible from either eye
                    left_eye_visible = (
                        abs(relative_angle_left) <= cfg.FISH_CONSTANTS["perception_field"]/2 and
                        np.linalg.norm(vector_from_left_eye) < cfg.AGENT_PARAMS["fish_detection_range"]
                    )
                    right_eye_visible = (
                        abs(relative_angle_right) <= cfg.FISH_CONSTANTS["perception_field"]/2 and
                        np.linalg.norm(vector_from_right_eye) < cfg.AGENT_PARAMS["fish_detection_range"]
                    )
                    
                    if left_eye_visible or right_eye_visible:
                        df.loc[idx, "walkerbot_visible"] = True
                        break
        
        feature_names.append("walkerbot_visible")

    return df, feature_names

def add_distance_angle_to_closest_walker(df, feature_names, add_binned_features=True):
    if "walkerbot_positions" in df.columns:
        feature_names.extend(["distance_to_closest_walker", "angle_to_closest_walker"])

        # Initialize columns
        df["distance_to_closest_walker"] = np.nan
        df["angle_to_closest_walker"] = np.nan

        for idx, row in df.iterrows():
            agent_position = np.array(row["position"])
            agent_orientation = row["orientation"]
            walkerbot_positions = row["walkerbot_positions"]

            if walkerbot_positions is None:
                continue

            walkerbot_positions = np.array(walkerbot_positions)
            if not pd.isna(walkerbot_positions).any() and walkerbot_positions.size > 0:
                walkerbot_positions = np.array(walkerbot_positions)
                
                # Calculate distances to all walkerbots
                distances = np.linalg.norm(walkerbot_positions - agent_position, axis=1)
                closest_idx = np.argmin(distances)
                
                # Distance to closest walker
                df.loc[idx, "distance_to_closest_walker"] = distances[closest_idx]
                
                # Angle to closest walker
                closest_walker_pos = walkerbot_positions[closest_idx]
                vector_to_walker = closest_walker_pos - agent_position
                angle_to_walker = np.arctan2(vector_to_walker[1], vector_to_walker[0]) - agent_orientation
                # Normalize to [-180, 180] degrees
                angle = (angle_to_walker + np.pi) % (2 * np.pi) - np.pi
                df.loc[idx, "angle_to_closest_walker"] = angle

    return df, feature_names


def add_distance_to_wall_circular(df, feature_names):
    # Make sure that arena_size is available on each row
    df = df.sort_values(by=["env_id", "episode_index", "time_step"])
    df["arena_size"] = df.groupby(["env_id", "episode_index"])["arena_size"].ffill()

    if {"position_x", "position_y", "arena_size"}.issubset(df.columns):
        try:
            df["distance_to_wall"] = df.apply(
                lambda row: (
                    row["arena_size"][0] / 2 - np.sqrt((row["position_x"] - row["arena_size"][0]/2)**2 + (row["position_y"] - row["arena_size"][0]/2)**2)
                    if not pd.isna(row["arena_size"])
                    else np.nan
                ),
                axis=1,
            )
            feature_names.append("distance_to_wall")
        except Exception as e:
            print(f"Could not calculate distance to wall: {e}")
            print(
                "Ensure 'position_x', 'position_y', and 'arena_size' are present in the DataFrame."
            )
    return df, feature_names

# Function to calculate distance to closest food
def distance_to_closest_food(row):
    agent_position = np.array(row["position"])
    food_positions = np.array(row["food_positions"])
    if food_positions.size == 0:
        return np.nan
    distances = np.linalg.norm(food_positions - agent_position, axis=1)
    return np.min(distances)


# Function to calculate angle to closest food
# TODO switch to radians, combine with distance_to_closest_food, or possibly vectorize
def angle_to_closest_food(row):
    agent_position = np.array(row["position"])
    agent_orientation = row["orientation"]
    food_positions = np.array(row["food_positions"])
    if food_positions.size == 0:
        return np.nan
    distances = np.linalg.norm(food_positions - agent_position, axis=1)
    closest_food_position = food_positions[np.argmin(distances)]
    vector_to_food = closest_food_position - agent_position
    angle = np.arctan2(vector_to_food[1], vector_to_food[0]) - agent_orientation
    angle_deg = np.degrees(angle) % 360
    # Convert from [0, 360] to [-180, 180]
    return (angle_deg + 180) % 360 - 180


def food_count_5cm(row):
    """
    Returns the total number of food items within 5 cm of the agent.
    """
    agent_position = np.array(row["position"])
    food_positions = np.array(row["food_positions"])

    if food_positions.size == 0:
        return 0  # No food at all

    # Distances from agent to each food item
    distances = np.linalg.norm(food_positions - agent_position, axis=1)

    # Count how many are within 5 cm
    return np.sum(distances <= 5.0)


def quadrant_food_count_5cm(row):
    """
    Returns a pd.Series with the count of food items in each quadrant
    (front, back, left, right), all within 5 cm of the agent.

    Quadrants are defined relative to agent_orientation = 0:
    - front: angle in [-45°, 45°)
    - right: angle in [-135°, -45°)
    - left: angle in [45°, 135°)
    - back: angle in [135°, 180) U [-180°, -135°)
    """
    agent_position = np.array(row["position"])
    agent_orientation = row["orientation"]  # in radians
    food_positions = np.array(row["food_positions"])

    # If no food, just return zero for all quadrants
    if food_positions.size == 0:
        return pd.Series(
            [0, 0, 0, 0],
            index=[
                "food_front_5cm",
                "food_back_5cm",
                "food_left_5cm",
                "food_right_5cm",
            ],
        )

    # Distances from agent to each food item
    distances = np.linalg.norm(food_positions - agent_position, axis=1)
    # Consider only food within 5 cm
    within_5_mask = distances <= 5.0
    relevant_food_positions = food_positions[within_5_mask]

    # If none within 5 cm, all quadrant counts are zero
    if len(relevant_food_positions) == 0:
        return pd.Series(
            [0, 0, 0, 0],
            index=[
                "food_front_5cm",
                "food_back_5cm",
                "food_left_5cm",
                "food_right_5cm",
            ],
        )

    # Vectors from agent to each food item (within 5 cm)
    vectors = relevant_food_positions - agent_position

    # Compute angles (relative to agent orientation)
    angles = np.arctan2(vectors[:, 1], vectors[:, 0]) - agent_orientation
    # Convert angles to degrees in [-180, 180]
    angles_deg = np.degrees(angles) % 360
    angles_deg = (angles_deg + 180) % 360 - 180

    # Define quadrant masks
    # front: [-45, 45)
    front_mask = (angles_deg >= -45) & (angles_deg < 45)
    # right: [-135, -45)
    right_mask = (angles_deg >= -135) & (angles_deg < -45)
    # left: [45, 135)
    left_mask = (angles_deg >= 45) & (angles_deg < 135)
    # back: everything else => [135, 180) or [-180, -135)
    back_mask = (angles_deg >= 135) | (angles_deg < -135)

    front_count = np.sum(front_mask)
    back_count = np.sum(back_mask)
    left_count = np.sum(left_mask)
    right_count = np.sum(right_mask)

    return pd.Series(
        [front_count, back_count, left_count, right_count],
        index=["food_front_5cm", "food_back_5cm", "food_left_5cm", "food_right_5cm"],
    )


def add_event_counters(df, event_columns):
    # Initialize columns for time since last event
    event_counter_colnames = []
    for event in event_columns:
        event_counter_colname = f"time_since_last_{event}"
        event_counter_colnames.append(event_counter_colname)
        df[event_counter_colname] = np.nan

    # Calculate time since last event with progress bar
    total_iterations = (
        len(df["env_id"].unique())
        * len(df["episode_index"].unique())
        * len(df["agent_id"].unique())
    )
    with tqdm.tqdm(total=total_iterations, desc="Processing events") as pbar:
        for env_id in df["env_id"].unique():
            for episode_index in df[df["env_id"] == env_id]["episode_index"].unique():
                for agent_id in df[
                    (df["env_id"] == env_id) & (df["episode_index"] == episode_index)
                ]["agent_id"].unique():
                    agent_df = df[
                        (df["env_id"] == env_id)
                        & (df["episode_index"] == episode_index)
                        & (df["agent_id"] == agent_id)
                    ]
                    for event in event_columns:
                        event_counter_colname = f"time_since_last_{event}"
                        last_time_step = None
                        for index, row in agent_df.iterrows():
                            if row[event]:
                                df.loc[index, event_counter_colname] = 0
                                last_time_step = row["time_step"]
                            elif last_time_step is not None:
                                df.loc[index, event_counter_colname] = (
                                    row["time_step"] - last_time_step
                                )
                    pbar.update(1)

    # Replace NaNs in time_since_last columns with a large number or -1 indicating no prior event
    # for event in event_columns:
    #     df[f"time_since_last_{event}"].fillna(-1, inplace=True)

    return df, event_counter_colnames


################ RNN related ################
def get_rnn_state_deltas(dff, pad_before=True):
    # first sort by (env_id, episode_index, agent_id, time_step)
    dff = dff.sort_values(
        by=["env_id", "episode_index", "agent_id", "time_step"], ignore_index=True
    )

    # For each (env_id, episode_index, agent_id), calculate delata_(rnn_state) per timestep
    def calculate_rnn_deltas(group):
        rnn_states = np.vstack(group["rnn_states"].tolist())
        deltas = np.diff(rnn_states, axis=0)
        deltas = np.linalg.norm(
            deltas, axis=1
        ).tolist()  # Calculate the norm of the deltas
        # Pad with zeros at the beginning to match original length
        if pad_before:
            deltas = [0.0] + deltas  # First timestep has no delta, so we pad with 0
        else:
            deltas = deltas + [0.0]
        assert len(deltas) == len(group["rnn_states"])
        return pd.Series(deltas, index=group.index)  # needed for transform()

    # Apply the function groupwise
    rnn_deltas = dff.groupby(
        ["env_id", "episode_index", "agent_id"], group_keys=False
    ).apply(calculate_rnn_deltas)
    return rnn_deltas


################ Attention related ################
def calculate_entropy(data_matrix):
    """
    Calculate the entropy of each row in a 2D numpy array.
    Each row represents attention weights for a specific time step.
    """
    if data_matrix.size == 0:
        return np.nan

    # Normalize the weights
    normalized_weights = data_matrix / np.sum(data_matrix, axis=1, keepdims=True)
    normalized_weights = np.nan_to_num(normalized_weights)  # Handle NaNs

    # Calculate entropy for each row
    entropy = -np.nansum(
        normalized_weights * np.log2(normalized_weights + 1e-10), axis=1
    )
    return entropy


def add_attention_entropy_columns(dff, sensor_types, colname="attn_mask"):
    """
    Add columns to dff for the entropy of attention weights for each sensor type.
    """
    data_matrix = np.stack(dff[colname].values)
    print("data_matrix.shape", data_matrix.shape)
    dff["attn_entropy"] = calculate_entropy(data_matrix)

    for sensor_name, indices in sensor_types:
        sensor_data = data_matrix[:, indices]
        col_name = f"attn_{sensor_name.lower()}_entropy"
        dff[col_name] = calculate_entropy(sensor_data)

    return dff


################ Uitls ################
def calculate_column_feature_correlations(dff, features, col="rnn_deltas"):
    """
    Calculate the correlation of a column with a list of features in a DataFrame.
    Args:
        dff (pd.DataFrame): DataFrame containing the data.
        features (list): List of feature names to calculate correlations with.
        col (str): The column name to correlate with the features.
    Returns:
        pd.DataFrame: DataFrame with features as index and their correlation with the specified column.
    """
    correlations = {}
    for idx in tqdm.tqdm(range(len(features)), desc="Calculating correlations"):
        feature = features[idx]
        if feature in dff.columns:
            corr = np.corrcoef(dff[col], dff[feature])[0, 1]
            correlations[feature] = corr
    return pd.DataFrame.from_dict(correlations, orient="index", columns=["corr"])
