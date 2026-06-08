from ast import Not
import unittest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cfg
from utils_figstyle import *
from matplotlib.patches import Circle
from matplotlib import patheffects as pe

def theil_index(consumptions, normalize = False):
    """Calculate Theil index for given consumptions."""
    n = len(consumptions) #n should be the total number of food eaten. Consumptions is a vector of integer values. Each entry is the amount of food eaten by each agent up until a given time point. n_agents x time points
    mean_consumption = np.mean(consumptions)
    if mean_consumption == 0:
        return 0

    consumptions = np.where(consumptions <= 0, 1e-12, consumptions) #avoid log(0)
    theil = (consumptions / mean_consumption) * np.log(consumptions / mean_consumption)
    if normalize:
        return (np.sum(theil) / n) / np.log(n)
    return np.sum(theil) / n


def calculate_theil_index_last(dff, normalize=False):
    """
    Calculate the Theil index for the last time step in each (env_id, episode_index).
    """
    # Filter eating events
    eating_events = dff[dff["eating_event"]].copy()
    agent_ids = dff["agent_id"].unique()

    # Calculate cumulative food consumed
    eating_events["cumulative_food"] = (
        eating_events.groupby(["env_id", "episode_index", "agent_id"]).cumcount() + 1
    )

    # Calculate Theil index at the last time step for each (env_id, episode_idx)
    results = []

    for (env_id, episode_index), subset in eating_events.groupby(
        ["env_id", "episode_index"]
    ):
        # Get the last time step
        last_time_step = subset["time_step"].max()
        # print(f"Env {env_id}, Episode {episode_index}, Last Time Step: {last_time_step}")
        subset_at_last_time = subset[subset["time_step"] <= last_time_step]
        consumptions = (
            subset_at_last_time.groupby("agent_id")["cumulative_food"].max().reindex(agent_ids, fill_value=0).values
        )
        # print(consumptions)
        theil = theil_index(consumptions, normalize=normalize)  # Custom function to calculate Theil index

        results.append(
            {
                "env_id": env_id,
                "episode_index": episode_index,
                "time_step": last_time_step,
                "theil_index": theil,
            }
        )

    theil_df = pd.DataFrame(results)
    theil_df.sort_values(by=["env_id", "episode_index", "time_step"], inplace=True)

    # Calculate aggregated Theil index over all data
    aggregated_consumptions = (
        eating_events.groupby("agent_id")["cumulative_food"].max().values
    )
    aggregated_theil = theil_index(aggregated_consumptions)

    # Return the Theil index DataFrame and the aggregated Theil index
    return theil_df, aggregated_theil

def calculate_theil_index_all(dff, normalize = False):
    """
    Calculate the Theil index for all time steps in each (env_id, episode_index).
    """
    # Filter eating events
    eating_events = dff[dff["eating_event"]].copy()
    agent_ids = dff["agent_id"].unique()

    # Calculate cumulative food consumed
    eating_events["cumulative_food"] = (
        eating_events.groupby(["env_id", "episode_index", "agent_id"]).cumcount() + 1
    )

    # Calculate Theil index for all time steps for each (env_id, episode_idx)
    results = []

    for (env_id, episode_index), subset in eating_events.groupby(
        ["env_id", "episode_index"]
    ):
        for time_step in subset["time_step"].unique():
            subset_at_time = subset[subset["time_step"] <= time_step]
            consumptions = (
                subset_at_time.groupby("agent_id")["cumulative_food"].max().reindex(agent_ids, fill_value=0).values
            )
            #print(consumptions)
            theil = theil_index(consumptions, normalize=normalize)  # Custom function to calculate Theil index
            results.append(
                {
                    "env_id": env_id,
                    "episode_index": episode_index,
                    "time_step": time_step,
                    "theil_index": theil,
                }
            )

        # # Get the last time step
        # last_time_step = subset["time_step"].max()
        # # print(f"Env {env_id}, Episode {episode_index}, Last Time Step: {last_time_step}")
        # subset_at_last_time = subset[subset["time_step"] <= last_time_step]
        # consumptions = (
        #     subset_at_last_time.groupby("agent_id")["cumulative_food"].max().values
        # )
        # # print(consumptions)
        # theil = theil_index(consumptions)  # Custom function to calculate Theil index

        # results.append(
        #     {
        #         "env_id": env_id,
        #         "episode_index": episode_index,
        #         "time_step": last_time_step,
        #         "theil_index": theil,
        #     }
        # )

    theil_df = pd.DataFrame(results)
    theil_df.sort_values(by=["env_id", "episode_index", "time_step"], inplace=True)

    # Calculate aggregated Theil index over all data
    aggregated_consumptions = (
        eating_events.groupby("agent_id")["cumulative_food"].max().values
    )
    aggregated_theil = theil_index(aggregated_consumptions)

    # Return the Theil index DataFrame and the aggregated Theil index
    return theil_df, aggregated_theil


def calculate_polarization(orientations):
    """
    Calculate the polarization of a group of orientations.

    Polarization is a measure of how aligned the orientations are within a group.
    It is calculated as the mean cosine of the difference between each orientation
    and the mean orientation of the group.

    Parameters:
    orientations (array-like): A list or array of orientation angles in radians.

    Returns:
    float: The polarization value, ranging from -1 (completely anti-aligned)
           to 1 (completely aligned).
    """
    orientations = np.array(orientations)
    mean_orientation = np.arctan2(
        np.mean(np.sin(orientations)), np.mean(np.cos(orientations))
    )
    return np.mean(np.cos(orientations - mean_orientation))


def calculate_cohesion(positions, sensing_radius = None):
    """
    Calculate the cohesion of a group of positions.

    Cohesion is measured as the average nearest neighbor distance within a group of positions.
    It indicates how close the individuals are to each other.

    Parameters:
    positions (array-like): A list or array of positions, where each position is
                            a list or array of coordinates [x, y].

    Returns:
    float: The average nearest neighbor distance. Returns NaN if there are
           fewer than 2 positions.
    """
    positions = np.array([pos for pos in positions])
    num_positions = len(positions)
    if num_positions < 2:
        return np.nan
    distances = np.linalg.norm(positions[:, np.newaxis] - positions, axis=2)
    np.fill_diagonal(distances, np.inf)
    nearest_distances = np.min(distances, axis=1)

    if sensing_radius is None:
        return np.mean(nearest_distances)

    # Filter out distances greater than the sensing radius
    valid_distances = nearest_distances[nearest_distances <= sensing_radius]
    
    # If there are no valid distances, return NaN
    if len(valid_distances) == 0:
        return np.nan
    return np.mean(valid_distances)

def calculate_cohesion_metrics(positions):
    """
    Calculate the cohesion, minimum, and maximum nearest-neighbor distances for a group of positions.

    Cohesion is measured as the average nearest neighbor distance within a group of positions.
    It indicates how close the individuals are to each other.

    Parameters:
    positions (array-like): A list or array of positions, where each position is
                            a list or array of coordinates [x, y].

    Returns:
    tuple: A tuple containing the average, minimum, and maximum nearest neighbor distances.
           Returns (NaN, NaN, NaN) if there are fewer than 2 positions.
    """
    positions = np.array([pos for pos in positions])
    num_positions = len(positions)
    if num_positions < 2:
        return np.nan, np.nan, np.nan

    # Calculate pairwise distances
    distances = np.linalg.norm(positions[:, np.newaxis] - positions, axis=2)
    np.fill_diagonal(
        distances, np.inf
    )  # Ignore self-distances by setting diagonal to infinity

    # Nearest neighbor distances
    nearest_distances = np.min(distances, axis=1)

    # Return average (cohesion), minimum, and maximum of nearest neighbor distances
    return (
        np.mean(nearest_distances),
        np.min(nearest_distances),
        np.max(nearest_distances),
    )

def calculate_pairwise_polarization(orientations, positions = None, sensing_radius = None):
    """
    Calculate the pairwise polarization of a group of orientations.

    Pairwise polarization is defined as the mean cosine of all pairwise differences
    between orientations.

    Parameters:
    orientations (array-like): A list or array of orientation angles in radians.

    Returns:
    float: The mean pairwise polarization value, ranging from -1 to 1.
    """
    orientations = np.array(orientations)

    if positions is not None and sensing_radius is not None:
        positions = np.array([pos for pos in positions])
        num_positions = len(positions)
        if num_positions != len(orientations):
            raise ValueError("Length of positions and orientations must be the same.")
        
        # Calculate pairwise distances
        distances = np.linalg.norm(positions[:, np.newaxis] - positions, axis=2)
        np.fill_diagonal(distances, np.inf)  # Ignore self-distances by setting diagonal to infinity
        
        # Create a mask for pairs within the sensing radius and i < j
        within_radius_mask = (distances <= sensing_radius) & np.triu(np.ones(distances.shape), k=1).astype(bool)
        
        # If no pairs are within the sensing radius, return NaN
        if not np.any(within_radius_mask):
            return np.nan
        
        # Calculate pairwise differences only for pairs within the sensing radius
        diffs = orientations[:, None] - orientations[None, :]
        cos_diffs = np.cos(diffs)
        
        # Extract the relevant pairwise cosines using the mask
        pairwise_cos = cos_diffs[within_radius_mask]
        
        return np.mean(pairwise_cos)

    N = orientations.size
    if N < 2:
        raise ValueError("Needs 2 agents or more to calculate pairwise polarization.")

    diffs = orientations[:, None] - orientations[None, :]
    cos_diffs = np.cos(diffs)

    # We exclude diagonal (self-self) and duplicate pairs (use i<j)
    i, j = np.triu_indices(N, k=1)
    pairwise_cos = cos_diffs[i, j]

    return np.mean(pairwise_cos)

def calculate_swimming_statistics(dff, sensing_radius=None):
    """
    Calculate swimming statistics for each group of agents at each time step.

    Parameters:
    dff (DataFrame): A pandas DataFrame containing columns 'env_id', 'episode_index', 
                     'time_step', 'position', and 'orientation'.

    Returns:
    DataFrame: A DataFrame containing the calculated polarization and cohesion for each
               env_id, episode_index, and time_step.
    """
    results = []
    for (env_id, episode_index, time_step), group in dff.groupby(["env_id", "episode_index", "time_step"]):
        group_positions = group["position"].apply(np.array).values
        group_orientations = group["orientation"].values
        polarization = calculate_polarization(group_orientations)
        cohesion = calculate_cohesion(group_positions, sensing_radius)
        pairwise_polarization = calculate_pairwise_polarization(group_orientations, group_positions, sensing_radius)
        results.append(
            {
                "env_id": env_id,
                "episode_index": episode_index,
                "time_step": time_step,
                "polarization": polarization,
                "cohesion": cohesion,
                "pairwise_polarization": pairwise_polarization,
            }
        )

    stats_df = pd.DataFrame(results).sort_values(by=["env_id", "episode_index", "time_step"])
    return stats_df

def calculate_swimming_statistics_by_agent(dff):
    """
    Calculate swimming statistics for each group of agents at each time step.

    Parameters:
    dff (DataFrame): A pandas DataFrame containing columns 'env_id', 'episode_index', 
                     'time_step', 'position', and 'orientation'.

    Returns:
    DataFrame: A DataFrame containing the calculated polarization and cohesion for each
               env_id, episode_index, and time_step.
    """
    results = []
    for (env_id, episode_index, time_step), group in dff.groupby(["env_id", "episode_index", "time_step"]):
        # Extract numpy arrays
        positions = np.vstack(group["position"].to_numpy())  # shape (N, 2)
        orientations = group["orientation"].to_numpy()       # shape (N,)
        agent_ids = group["agent_id"].to_numpy()

        # --- Compute pairwise distances ---
        diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=-1)

        # Ignore self-distances
        np.fill_diagonal(dists, np.inf)

        # --- Find closest agent for each ---
        closest_indices = np.argmin(dists, axis=1)
        closest_agents = agent_ids[closest_indices]
        closest_distances = dists[np.arange(len(agent_ids)), closest_indices]

        # --- Compute orientation difference (angle between headings) ---
        closest_orientations = orientations[closest_indices]
        orientation_diffs = orientations - closest_orientations

        # Wrap differences to [-pi, pi] for correctness
        orientation_diffs = (orientation_diffs + np.pi) % (2 * np.pi) - np.pi

        # Cosine of angle difference = alignment measure
        nn_cosine_orientation = np.cos(orientation_diffs)

        
        # --- Build one record per agent ---
        df_temp = pd.DataFrame({
            "env_id": env_id,
            "episode_index": episode_index,
            "time_step": time_step,
            "agent_id": agent_ids,
            "closest_agent_id": closest_agents,
            "nn_distance": closest_distances,
            "nn_cosine_orientation": nn_cosine_orientation,
        })

        results.append(df_temp)

    stats_df = pd.concat(results).sort_values(by=["env_id", "episode_index", "time_step", "agent_id"]).reset_index()
    return stats_df



def calculate_fluctuation_ratio(displacements):
    """
    Calculate the fluctuation ratio r = <x^2> / <x>^2.

    Parameters:
    displacements (array-like): List of x displacements (or y, depending on choice).

    Returns:
    float: Fluctuation ratio
    """
    displacements = np.array(displacements)
    mean_x = np.mean(displacements)
    mean_x2 = np.mean(displacements**2)
    if mean_x == 0:
        return np.nan
    return mean_x2 / (mean_x**2)


def calculate_hurst_exponent(ts):
    """
    Estimate the Hurst exponent of a time series.

    Parameters:
    ts (array-like): The time series (e.g., angle positions)

    Returns:
    float: Hurst exponent estimate

    """
    ts = np.array(ts)
    N = len(ts)
    if N < 20:
        return np.nan  # Not enough data

    T = np.arange(1, N + 1)
    Y = np.cumsum(ts - np.mean(ts))
    R = np.max(Y) - np.min(Y)
    S = np.std(ts)
    if S == 0:
        return 0.0
    return np.log(R / S) / np.log(N)


############ TRAJECTORY ANALYSIS ############


def plot_agent_trajectories_for_specific_episode(dff, env_id, episode_idx):
    subset = dff[(dff["env_id"] == env_id) & (dff["episode_index"] == episode_idx)]
    agent_ids = subset["agent_id"].unique()

    fig, axes = plt.subplots(1, len(agent_ids), figsize=(12, 3))
    for i, agent_id in enumerate(agent_ids):
        agent_data = subset[subset["agent_id"] == agent_id]
        positions = np.array(agent_data["position"].tolist())

        # Plotting trajectories
        ax = axes[i]
        ax.plot(positions[:, 0], positions[:, 1], label=f"Agent {agent_id}")

        ax.set_title(f"Env {env_id}, Episode {episode_idx}, Agent {agent_id}")
        ax.set_xlabel("X Position")
        ax.set_ylabel("Y Position")
        ax.grid(True)
    #         ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.show()


def plot_agent_trajectories_with_eating_events(dff):
    unique_envs = dff["env_id"].unique()
    unique_episodes = dff["episode_index"].unique()

    for env_id in unique_envs:
        for episode_idx in unique_episodes:
            subset = dff[
                (dff["env_id"] == env_id) & (dff["episode_index"] == episode_idx)
            ]
            plt.figure(figsize=(4, 4))
            agent_ids = subset["agent_id"].unique()
            for agent_id in agent_ids:
                agent_data = subset[subset["agent_id"] == agent_id]
                positions = np.array(agent_data["position"].tolist())
                plt.plot(positions[:, 0], positions[:, 1], label=f"Agent {agent_id}")
                eating_positions = np.array(
                    agent_data[agent_data["eating_event"]]["position"].tolist()
                )
                if len(eating_positions) > 0:
                    plt.scatter(
                        eating_positions[:, 0],
                        eating_positions[:, 1],
                        c="red",
                        marker="x",
                        label=f"Eating" if agent_id == agent_ids[-1] else None,
                    )
            plt.title(f"Env {env_id}, Episode {episode_idx}")
            plt.xlabel("X Position")
            plt.ylabel("Y Position")
            plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
            plt.grid(True)
            plt.show()

def calculate_move_turn_stats(dff):
    """
    Calculate basic statistics for move_forward and turn_angle columns.
    """
    import scipy.stats

    import numpy as np
    from scipy.stats import gaussian_kde
    from scipy.signal import find_peaks
    import matplotlib.pyplot as plt

    # KDE estimate
    move_forward = dff["move_forward"]
    kde = gaussian_kde(move_forward)
    x = np.linspace(min(move_forward)-1, max(move_forward)+1, 500)
    y = kde(x)

    peaks, _ = find_peaks(y)

    # Calculate statistics for move_forward
    move_stats = {
        "mean": np.mean(move_forward),
        "variance": np.var(move_forward),
        "entropy": scipy.stats.entropy(np.histogram(move_forward, bins=50)[0] + 1e-10),  # Add small value to avoid log(0)
        "median": np.median(move_forward),
        "skewness": scipy.stats.skew(move_forward, bias=False),
        "kurtosis": scipy.stats.kurtosis(move_forward, bias=False),
        "mode_values": x[peaks].tolist(),
    }
    
    # KDE estimate
    turn_angle = dff["turn_angle"]
    kde = gaussian_kde(turn_angle)
    x = np.linspace(min(turn_angle)-1, max(turn_angle)+1, 500)
    y = kde(x)

    peaks, _ = find_peaks(y)

    # Calculate statistics for turn_angle
    turn_stats = {
        "mean": np.mean(turn_angle),
        "variance": np.var(turn_angle),
        "entropy": scipy.stats.entropy(np.histogram(turn_angle, bins=50)[0] + 1e-10),  # Add small value to avoid log(0)
        "median": np.median(turn_angle),
        "skewness": scipy.stats.skew(turn_angle, bias=False),
        "kurtosis": scipy.stats.kurtosis(turn_angle, bias=False),
        "mode_values": x[peaks].tolist(),
    }

    return move_stats, turn_stats

def plot_first_success_story(dff, first_success, is_miss=False, save_path=None, dpi=300, prey_pre_dx_dy=(-67, -18), prey_during_dx_dy=(-80, -20)):
    """
    Plots a single successful hunt as a mini-story:
      - blue = exploring just before detection
      - orange = hunting after detection until capture
      - green = prey path during hunt
      - red = prey path in the moments before the hunt starts
      - start / capture points called out explicitly
    """
    set_nature_style()

    # --- pick the first success (already sorted upstream typically) ---

    env_id         = first_success["env_id"]
    episode_index  = first_success["episode_index"]
    agent_id       = first_success["agent_id"]
    food_id        = first_success["food_id"]
    start_time     = int(first_success["start_time_step"])
    duration       = int(first_success["tracking_duration"])

    # time windows
    explore_start  = max(0, start_time - 20)
    explore_end    = start_time
    hunt_end       = start_time + duration

    # --- gather slices ---
    traj = dff[(dff["env_id"]==env_id) &
               (dff["episode_index"]==episode_index) &
               (dff["agent_id"]==agent_id)].sort_values("time_step")

    exploring = traj[(traj["time_step"] >= explore_start) & (traj["time_step"] <= explore_end)]
    hunting   = traj[(traj["time_step"] >= explore_end)   & (traj["time_step"] <= hunt_end)]

    # positions (agent)
    if exploring.empty or hunting.empty:
        print("Warning: exploring or hunting window empty for selected success; skipping.")
        return

    exp_pos = np.vstack(exploring["position"].to_numpy())
    hunt_pos = np.vstack(hunting["position"].to_numpy())

    # prey positions filtered to the tracked food_id
    def _extract_food_positions(df):
        pts = []
        for positions, fids in zip(df["food_positions"], df["food_ids"]):
            if food_id in fids:
                idx = fids.index(food_id)
                pts.append(positions[idx])
        return np.array(pts) if len(pts) else np.empty((0,2))

    prey_pre  = _extract_food_positions(exploring)
    prey_during = _extract_food_positions(hunting)

    # --- arena geometry ---
    arena_tuple = dff[(dff["episode_index"]==episode_index) & (dff["env_id"]==env_id)].iloc[0]["arena_size"]
    arena_diam  = arena_tuple[0] if isinstance(arena_tuple, tuple) else float(arena_tuple)
    arena_rad   = arena_diam / 2.0
    arena_ctr   = (arena_rad, arena_rad)

    # --- style defaults (compact, colorblind safe) ---
    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "savefig.dpi": dpi,
        "pdf.fonttype": 42,   # editable text in vector editors
        "ps.fonttype": 42
    })

    # Okabe–Ito palette
    c_explore = "#0072B2"   # blue
    c_hunt    = "#E69F00"   # orange
    c_prey    = "#009E73"   # green
    c_preypre = "#D55E00"   # red/orange
    c_arena   = "#9e9e9e"

    fig, ax = plt.subplots(figsize=(4, 4))  # square for equal aspect

    # trajectories
    ax.plot(exp_pos[:,0],  exp_pos[:,1],  lw=2.0, color=c_explore, label="Exploring (pre‑detect)",
            path_effects=[pe.Stroke(linewidth=2.8, foreground="white"), pe.Normal()])
    ax.plot(hunt_pos[:,0], hunt_pos[:,1], lw=2.2, color=c_hunt, label="Hunting (lock‑on → strike)",
            path_effects=[pe.Stroke(linewidth=3.0, foreground="white"), pe.Normal()])

    # direction cues (sparse arrows along hunting path)
    # if len(hunt_pos) > 4:
    #     step = max(1, len(hunt_pos)//12)  # ~12 arrows max
    #     seg   = hunt_pos[::step]
    #     dxy   = np.diff(seg, axis=0)
    #     ax.quiver(seg[:-1,0], seg[:-1,1], dxy[:,0], dxy[:,1],
    #               angles='xy', scale_units='xy', scale=1.0, width=0.003, alpha=0.7, color=c_hunt, zorder=3)

    # prey paths
    if prey_during.size:
        ax.plot(prey_during[:,0], prey_during[:,1], lw=1.6, color=c_prey, alpha=0.9, label="Prey (during hunt)")
        ax.scatter(prey_during[-1,0], prey_during[-1,1], s=28, color=c_prey, zorder=4)
        ax.annotate("Prey path\n(during hunt)",
                    xy=(prey_during[-1,0], prey_during[-1,1]), xytext=prey_during_dx_dy,
                    textcoords="offset points", fontsize=12,
                    color=c_prey)
    if prey_pre.size:
        ax.plot(prey_pre[:,0], prey_pre[:,1], lw=1.2, color=c_preypre, alpha=0.8, label="Prey (pre‑hunt)")
        ax.annotate("Prey path\n(pre‑hunt)",
                xy=(prey_pre[0,0], prey_pre[0,1]), xytext=prey_pre_dx_dy,
                textcoords="offset points", fontsize=12,
                color=c_preypre)

    # key moments
    ax.scatter(exp_pos[0,0], exp_pos[0,1], s=36, marker="o", color=c_explore, zorder=5)
    start = (explore_start - explore_end) / cfg.ENV_PARAMS["fps_sim"]
    ax.annotate(f"Explore segment\n({start:.1f}→0 s)",
                xy=(exp_pos[0,0], exp_pos[0,1]), xytext=(-14, 14),
                textcoords="offset points", fontsize=12,
                arrowprops=dict(arrowstyle="->", lw=1.0, color=c_explore), color=c_explore)

    ax.scatter(hunt_pos[0,0], hunt_pos[0,1], s=46, marker="*", color=c_hunt, zorder=6)
    ax.annotate("Detect & begin hunt\n(t = 0 s)",
                xy=(hunt_pos[0,0], hunt_pos[0,1]), xytext=(10, -18),
                textcoords="offset points", fontsize=12,
                arrowprops=dict(arrowstyle="->", lw=1.0, color=c_hunt), color=c_hunt)

    
    if is_miss:
        ax.scatter(hunt_pos[-1,0], hunt_pos[-1,1], s=60, marker="X", color="red", zorder=7)
        ax.annotate("Abort\n(t = {:.1f} s)".format(duration/ cfg.ENV_PARAMS["fps_sim"]),
                xy=(hunt_pos[-1,0], hunt_pos[-1,1]), xytext=(8, 10),
                textcoords="offset points", fontsize=12,
                arrowprops=dict(arrowstyle="->", lw=1.0, color="black"), color="black")
    else:
        ax.scatter(hunt_pos[-1,0], hunt_pos[-1,1], s=60, marker="X", color="black", zorder=7)
        ax.annotate("Strike\n(t = {:.1f} s)".format(duration/ cfg.ENV_PARAMS["fps_sim"]),
                xy=(hunt_pos[-1,0], hunt_pos[-1,1]), xytext=(8, 10),
                textcoords="offset points", fontsize=12,
                arrowprops=dict(arrowstyle="->", lw=1.0, color="black"), color="black")

    all_points = np.vstack([
        exp_pos,
        hunt_pos,
        prey_pre if prey_pre.size else np.empty((0, 2)),
        prey_during if prey_during.size else np.empty((0, 2))
    ])

    # Bounding box
    xmin, ymin = np.min(all_points, axis=0)
    xmax, ymax = np.max(all_points, axis=0)

    # Add padding (e.g., 30% of the range)
    xpad = (xmax - xmin) * 0.3
    ypad = (ymax - ymin) * 0.3
    xmin -= xpad
    xmax += xpad
    ymin -= ypad
    ymax += ypad

    # Keep aspect ratio square
    width = xmax - xmin
    height = ymax - ymin
    side = max(width, height)
    xc = (xmin + xmax) / 2
    yc = (ymin + ymax) / 2
    xmin, xmax = xc - side / 2, xc + side / 2
    ymin, ymax = yc - side / 2, yc + side / 2

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    # Keep arena boundary for context (faded)
    arena = Circle(arena_ctr, arena_rad, fill=False, lw=1.0, ls=(0, (3,2)), ec=c_arena, alpha=0.4, zorder=0)
    ax.add_patch(arena)

    # labels
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    # ax.set_title("First Successful Hunt: Explore → Lock‑on → Capture", pad=10)

    # legend (compact, outside)
    # leg = ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2, handlelength=2.5)
    # for lh in leg.legendHandles:
    #     lh.set_alpha(1.0)

    # minimal spines, subtle grid
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.grid(True, linewidth=0.6, alpha=0.2)

    sns.despine(ax=ax)

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, format="svg")
        fig.savefig(save_path.replace(".svg", ".png"), format="png", dpi=dpi)
    return fig, ax

########### AROUND EATING ANALYSIS ###########

def plot_behavioral_maneuvers_around_eating(dff, results_folder):
    agents = dff["agent_id"].unique()
    window_size = 5

    for agent_id in agents:
        agent_data = dff[dff["agent_id"] == agent_id]

        speed_changes = []
        trajectory_curvatures = []
        vergence_angles = []
        left_eye_angles = []
        right_eye_angles = []

        for (env_id, episode_index), subset_data in agent_data.groupby(
            ["env_id", "episode_index"]
        ):
            eating_indices = subset_data[subset_data["eating_event"]].index

            for idx in eating_indices:
                pos = subset_data.index.get_loc(idx)
                if pos - window_size >= 0 and pos + window_size < len(subset_data):
                    segment = subset_data.iloc[
                        pos - window_size : pos + window_size + 1
                    ]

                    # print(segment["displacement"].values)

                    # Calculate speed change
                    speed_change = segment["displacement"].values * cfg.ENV_PARAMS["fps_sim"]
                    speed_changes.append(speed_change)

                    # Calculate trajectory curvature
                    positions = np.stack(segment["position"].values)
                    diffs = np.diff(positions, axis=0)
                    directions = np.arctan2(diffs[:, 1], diffs[:, 0])
                    curvature = np.abs(np.diff(directions)) * cfg.ENV_PARAMS["fps_sim"]
                    curvature = np.concatenate(
                        ([0], curvature, [0])
                    )  # to match the length of the window
                    trajectory_curvatures.append(curvature)

                    # Extract vergence angle
                    vergence_angle = segment["vergence_angle_deg"].values
                    vergence_angles.append(vergence_angle)

                    left_eye_angle = segment["left_eye_angle"].values
                    left_eye_angles.append(left_eye_angle)

                    right_eye_angle = segment["right_eye_angle"].values
                    right_eye_angles.append(right_eye_angle)

        if (
            speed_changes and trajectory_curvatures and vergence_angles and left_eye_angles and right_eye_angles
        ):  # Check if there are any valid maneuvers
            speed_changes = np.array(speed_changes)
            trajectory_curvatures = np.array(trajectory_curvatures)
            vergence_angles = np.array(vergence_angles)
            left_eye_angles = np.array(left_eye_angles)
            right_eye_angles = np.array(right_eye_angles)

            # Calculate mean and standard error for speed changes
            mean_speed_change = np.mean(speed_changes, axis=0)
            std_speed_change = np.std(speed_changes, axis=0) # / np.sqrt(speed_changes.shape[0])

            # Calculate mean and standard error for trajectory curvatures
            mean_curvature = np.mean(trajectory_curvatures, axis=0)
            std_curvature = np.std(trajectory_curvatures, axis=0) # / np.sqrt(trajectory_curvatures.shape[0])

            # Calculate mean and standard error for vergence angles
            mean_vergence = np.mean(vergence_angles, axis=0)
            std_vergence = np.std(vergence_angles, axis=0) # / np.sqrt(vergence_angles.shape[0])

            mean_left_eye_angle = np.mean(left_eye_angles, axis=0)
            std_left_eye_angle = np.std(left_eye_angles, axis=0) # / np

            mean_right_eye_angle = np.mean(right_eye_angles, axis=0)
            std_right_eye_angle = np.std(right_eye_angles, axis=0) # / np

            print(f"Agent {agent_id} speed_changes.shape", speed_changes.shape)
            print(
                f"Agent {agent_id} trajectory_curvatures.shape",
                trajectory_curvatures.shape,
            )
            print(f"Agent {agent_id} vergence_angles.shape", vergence_angles.shape)

            time_steps = np.arange(-window_size, window_size + 1)

            # Plot speed changes
            plt.figure(figsize=(5, 2))
            plt.plot(time_steps, mean_speed_change, label="Speed Change")
            plt.fill_between(
                time_steps,
                mean_speed_change - std_speed_change,
                mean_speed_change + std_speed_change,
                alpha=0.2,
            )
            plt.axvline(x=0, color="red", linestyle="--", label="Eating Event")
            plt.title(f"Agent {agent_id} Speed Changes Around Eating Events")
            plt.xlabel("Time Steps")
            plt.ylabel("Speed (mm/s)")
            plt.legend()
            plt.savefig(f"{results_folder}/agent_{agent_id}_speed_changes.png", dpi=300)
            plt.show()

            # Plot trajectory curvatures
            plt.figure(figsize=(5, 2))
            plt.plot(time_steps, mean_curvature, label="Trajectory Curvature")
            plt.fill_between(
                time_steps,
                mean_curvature - std_curvature,
                mean_curvature + std_curvature,
                alpha=0.2,
            )
            plt.axvline(x=0, color="red", linestyle="--", label="Eating Event")
            plt.title(f"Agent {agent_id} Trajectory Curvature Around Eating Events")
            plt.xlabel("Time Steps")
            plt.ylabel("Curvature (rad/s)")
            plt.legend()
            plt.savefig(f"{results_folder}/agent_{agent_id}_trajectory_curvature.png", dpi=300)
            plt.show()

            # Plot vergence angles
            plt.figure(figsize=(5, 2))
            plt.plot(time_steps, mean_vergence, label="Vergence Angle")
            plt.fill_between(
                time_steps,
                mean_vergence - std_vergence,
                mean_vergence + std_vergence,
                alpha=0.2,
            )
            plt.axvline(x=0, color="red", linestyle="--", label="Eating Event")
            plt.title(f"Agent {agent_id} Vergence Angle Around Eating Events")
            plt.xlabel("Time Steps")
            plt.ylabel("Vergence Angle (degrees)")
            plt.legend()
            plt.savefig(f"{results_folder}/agent_{agent_id}_vergence_angle.png", dpi=300)
            plt.show()

            plt.figure(figsize=(5, 2))
            plt.plot(time_steps, mean_left_eye_angle, label="Left Eye Angle")
            plt.fill_between(
                time_steps,
                mean_left_eye_angle - std_left_eye_angle,
                mean_left_eye_angle + std_left_eye_angle,
                alpha=0.2,
            )
            plt.axvline(x=0, color="red", linestyle="--", label="Eating Event")
            plt.title(f"Agent {agent_id} Left Eye Angle Around Eating Events")
            plt.xlabel("Time Steps")
            plt.ylabel("Left Eye Angle (radians)")
            plt.legend()
            plt.savefig(f"{results_folder}/agent_{agent_id}_left_eye_angle.png", dpi=300)
            plt.show()

            plt.figure(figsize=(5, 2))
            plt.plot(time_steps, mean_right_eye_angle, label="Right Eye Angle")
            plt.fill_between(
                time_steps,
                mean_right_eye_angle - std_right_eye_angle,
                mean_right_eye_angle + std_right_eye_angle,
                alpha=0.2,
            )
            plt.axvline(x=0, color="red", linestyle="--", label="Eating Event")
            plt.title(f"Agent {agent_id} Right Eye Angle Around Eating Events")
            plt.xlabel("Time Steps")
            plt.ylabel("Right Eye Angle (radians)")
            plt.legend()
            plt.savefig(f"{results_folder}/agent_{agent_id}_right_eye_angle.png", dpi=300)
            plt.show()

def analyze_distance_angle_to_consumed_food(dff, window_size=20):
    """
    Analyze distance and angle to consumed food before eating events.
    """
    
    distance_angle_data = []
    
    # Group by agent and episode
    for (env_id, episode_index, agent_id), agent_data in dff.groupby(['env_id', 'episode_index', 'agent_id']):
        agent_data = agent_data.sort_values('time_step').reset_index(drop=True)
        
        # Find eating events
        eating_indices = agent_data[agent_data['eating_event']].index
        
        for eating_idx in eating_indices:
            if eating_idx < window_size:
                continue  # Not enough data before eating event
            
            # Get window before eating event (including the eating moment)
            window_data = agent_data.iloc[eating_idx - window_size:eating_idx + 1]
            
            # Get the food item that was eaten by comparing eaten_food_ids before and after
            eating_row = agent_data.iloc[eating_idx]
            pre_eating_row = agent_data.iloc[eating_idx - 1]
            
            # Find which food was eaten by comparing eaten_food_ids
            eaten_before = set(pre_eating_row['eaten_food_ids'])
            eaten_after = set(eating_row['eaten_food_ids'])
            newly_eaten_ids = eaten_after - eaten_before
            
            if not newly_eaten_ids:
                print(eaten_before, eaten_after)
                continue  # No new food eaten, skip this event
            
            # Get the ID of the consumed food (should be only one)
            consumed_food_id = list(newly_eaten_ids)[0]

            food_ids = pre_eating_row['food_ids']
            
            for i, food_id in enumerate(food_ids):
                if food_id == consumed_food_id:
                    target_food_pos = pre_eating_row['food_positions'][i]
                    break

            # Calculate distance and angle to target food for each time step in the window
            for i, (_, row) in enumerate(window_data.iterrows()):
                agent_pos_t = np.array(row['position'])
                agent_orientation = row['orientation']
                
                # Distance to target food
                distance_to_target = np.linalg.norm(target_food_pos - agent_pos_t)
                
                # Direction from agent to target food
                food_vector = target_food_pos - agent_pos_t
                food_direction = np.arctan2(food_vector[1], food_vector[0])
                
                # Calculate angular difference (angle to food relative to agent orientation)
                angular_diff = food_direction - agent_orientation
                angular_diff = np.arctan2(np.sin(angular_diff), np.cos(angular_diff))
                angular_diff_deg = angular_diff * 180 / np.pi
                
                # Store data for this time step
                distance_angle_data.append({
                    'env_id': env_id,
                    'episode_index': episode_index,
                    'agent_id': agent_id,
                    'eating_time_step': eating_row['time_step'],
                    'time_relative': i - window_size,  # Negative values before eating, 0 at eating
                    'distance_to_consumed_food': distance_to_target,
                    'angle_to_consumed_food_deg': angular_diff_deg
                })
    
    return pd.DataFrame(distance_angle_data)


########### EYE VERGENCE / HUNTING ANALYSIS ###########

def analyze_vergence_during_food_tracking(dff):
    """
    Identify non-overlapping food tracking sequences per agent. 
    A sequence starts when food is first detected.
    Ends with:
      - Success: food is eaten.
      - Miss: food leaves nearby_food_ids without being eaten.
    Tracking duration is computed in time_step units.
    Sequences of zero duration are dropped.
    """
    tracking_data = []

    # Group by agent and episode
    for (env_id, episode_index, agent_id), agent_data in (
        dff.groupby(['env_id', 'episode_index', 'agent_id'])
    ):
        agent_data = agent_data.sort_values('time_step').reset_index(drop=True)

        # Ongoing tracks: food_id → metadata
        active = {}

        for _, row in agent_data.iterrows():
            t = row['time_step']
            detected = set(row['detected_food_ids'])
            binocular = set(row['binocular_food_ids'])
            nearby   = set(row['nearby_food_ids'])
            eaten    = set(row['eaten_food_ids'])
            vergence = row['vergence_angle_deg']
            speed    = row['speed']

            # 1) Start new tracks
            for fid in detected:
                if fid not in active:
                    active[fid] = {
                        'start_time': t,
                        'vergence_angles': [],
                        'speeds': [],
                        'detected': [],
                        'binocular': []
                    }

            # 2) Update existing tracks
            for fid in list(active):
                track = active[fid]
                track['vergence_angles'].append(vergence)
                track['speeds'].append(speed)
                track['binocular'].append(fid in binocular)
                track['detected'].append(fid in detected)

                # Check for end conditions
                if fid in eaten:
                    outcome = 'success'
                elif fid not in nearby:
                    outcome = 'miss'
                else:
                    continue  # still tracking

                # Compute duration in time steps
                duration = t - track['start_time']
                # Only keep if we actually saw it for at least one full step
                if duration > 0:
                    tracking_data.append({
                        'env_id': env_id,
                        'episode_index': episode_index,
                        'agent_id': agent_id,
                        'food_id': fid,
                        'start_time_step': track['start_time'],
                        'tracking_duration': duration,
                        'outcome': outcome,
                        'vergence_angles': track['vergence_angles'],
                        'mean_vergence_angle': np.mean(track['vergence_angles']),
                        'std_vergence_angle': np.std(track['vergence_angles']),
                        'initial_vergence_angle': track['vergence_angles'][0],
                        'final_vergence_angle': track['vergence_angles'][-1],
                        'speeds': track['speeds'],
                        'mean_speed': np.mean(track['speeds']),
                        'std_speed': np.std(track['speeds']),
                        'binocular': track['binocular'],
                        'binocular_frequency': np.mean(track['binocular']),
                        'detected': track['detected'],
                        'detected_frequency': np.mean(track['detected'])
                    })

                # Remove it from active, whether we recorded it or not
                del active[fid]

        # Optional: flush unfinished tracks as misses at end of episode
        # last_t = agent_data['time_step'].iloc[-1]
        # for fid, track in active.items():
        #     duration = last_t - track['start_time']
        #     if duration > 0:
        #         tracking_data.append({
        #             'env_id': env_id,
        #             'episode_index': episode_index,
        #             'agent_id': agent_id,
        #             'food_id': fid,
        #             'start_time_step': track['start_time'],
        #             'tracking_duration': duration,
        #             'outcome': 'miss',
        #             'vergence_angles': track['vergence_angles'],
        #             'mean_vergence_angle': np.mean(track['vergence_angles']),
        #             'std_vergence_angle': np.std(track['vergence_angles']),
        #             'initial_vergence_angle': track['vergence_angles'][0],
        #             'final_vergence_angle': track['vergence_angles'][-1],
        #             'speeds': track['speeds'],
        #             'mean_speed': np.mean(track['speeds']),
        #             'std_speed': np.std(track['speeds'])
        #         })

    return pd.DataFrame(tracking_data)


# Calculate average vergence angles for each time step relative to outcome
def calculate_avg_vergence_by_outcome(tracking_results):
    """Calculate average vergence angles by outcome and relative time step"""
    success_data = tracking_results[tracking_results['outcome'] == 'success']
    miss_data = tracking_results[tracking_results['outcome'] == 'miss']
    
    # Get maximum tracking duration to normalize time steps
    max_duration = tracking_results['tracking_duration'].max()
    
    success_trajectories = []
    miss_trajectories = []
    
    # Process success trajectories
    for _, row in success_data.iterrows():
        vergence_angles = row['vergence_angles']
        duration = row['tracking_duration']
        
        # Normalize to relative time (0 = start, 1 = end)
        if duration > 1:
            time_points = np.linspace(0, 1, len(vergence_angles))
            success_trajectories.append((time_points, vergence_angles))
    
    # Process miss trajectories
    for _, row in miss_data.iterrows():
        vergence_angles = row['vergence_angles']
        duration = row['tracking_duration']
        
        # Normalize to relative time (0 = start, 1 = end)
        if duration > 1:
            time_points = np.linspace(0, 1, len(vergence_angles))
            miss_trajectories.append((time_points, vergence_angles))
    
    return success_trajectories, miss_trajectories

def interpolate_trajectories(success_trajectories, miss_trajectories, num_points=50):
    # Create interpolated data on common time grid
    time_grid = np.linspace(0, 1, num_points)  # 50 time points from start to end

    # Interpolate success trajectories
    success_interpolated = []
    avg_auc = 0
    for time_points, angles in success_trajectories:
        if len(time_points) > 1:
            interpolated = np.interp(time_grid, time_points, angles)
            # Calculate area under curve for this trajectory
            success_interpolated.append(interpolated)

    # Interpolate miss trajectories
    miss_interpolated = []
    for time_points, angles in miss_trajectories:
        if len(time_points) > 1:
            interpolated = np.interp(time_grid, time_points, angles)
            miss_interpolated.append(interpolated)

    return success_interpolated, miss_interpolated, time_grid

def non_tracking_data(dff, tracking_results):
    # Find times when agent is not in tracking sequences
    non_tracking_data = []

    for (env_id, episode_index, agent_id), agent_data in dff.groupby(['env_id', 'episode_index', 'agent_id']):
        agent_data = agent_data.sort_values('time_step').reset_index(drop=True)
        
        # Get all time steps when agent was tracking food
        tracking_times = set()
        agent_tracking = tracking_results[
            (tracking_results['env_id'] == env_id) & 
            (tracking_results['episode_index'] == episode_index) & 
            (tracking_results['agent_id'] == agent_id)
        ]
        
        for _, track in agent_tracking.iterrows():
            start_time = track['start_time_step']
            duration = track['tracking_duration']
            tracking_times.update(range(start_time, start_time + duration + 1))
        
        # Collect vergence angles for non-tracking times
        non_tracking_vergence = []
        for _, row in agent_data.iterrows():
            if row['time_step'] not in tracking_times:
                non_tracking_vergence.append(row['vergence_angle_deg'])
        
        if non_tracking_vergence:
            non_tracking_data.extend(non_tracking_vergence)

    return non_tracking_data

def calculate_auc_hunting_no_hunting(success_trajectories, non_tracking_data, std_info=False):
    avg_auc = 0
    count = 0
    aucs = []
    for time_points, angles in success_trajectories:
        if len(time_points) > 1:
            # Calculate area under curve for success trajectories
            max_vergence = (cfg.FISH_CONSTANTS["max_left_vergence"] - cfg.FISH_CONSTANTS["max_right_vergence"] + cfg.FISH_CONSTANTS["perception_field"]) * 180 / np.pi
            min_vergence = (cfg.FISH_CONSTANTS["min_left_vergence"] - cfg.FISH_CONSTANTS["min_right_vergence"] + cfg.FISH_CONSTANTS["perception_field"]) * 180 / np.pi
            auc = np.trapz(np.array(angles) - min_vergence, time_points) / ((max_vergence - min_vergence) * (np.max(time_points) - np.min(time_points))) if (max_vergence - min_vergence) != 0 else 0
            # aucs.append(auc)
            avg_auc += auc
            count += 1

    avg_auc /= count if count > 0 else 1
    # std_auc = np.std(aucs) if count > 0 else 0

    # Calculate AUC for non-tracking periods
    if non_tracking_data:
        max_vergence = (cfg.FISH_CONSTANTS["max_left_vergence"] - cfg.FISH_CONSTANTS["max_right_vergence"] + cfg.FISH_CONSTANTS["perception_field"]) * 180 / np.pi
        min_vergence = (cfg.FISH_CONSTANTS["min_left_vergence"] - cfg.FISH_CONSTANTS["min_right_vergence"] + cfg.FISH_CONSTANTS["perception_field"]) * 180 / np.pi
    
        # Create time grid for non-tracking data (assuming uniform sampling)
        time_grid_non_tracking = np.linspace(0, 1, len(non_tracking_data))
    
        non_tracking_auc = np.trapz(np.array(non_tracking_data) - min_vergence, time_grid_non_tracking) / (max_vergence - min_vergence) if (max_vergence - min_vergence) != 0 else 0  

    if std_info:
        return avg_auc, non_tracking_auc #, std_auc, non_tracking_std_auc
    else:
        return avg_auc, non_tracking_auc

def time_to_convergence(success_trajectories):
    # For each success trajectory, compute the time it takes before eyes are converged
    convergence_times = []

    # Define convergence threshold (close to the convergence state)
    convergence_threshold = 5  # radians from convergence state
    max_vergence = (cfg.FISH_CONSTANTS["max_left_vergence"] - cfg.FISH_CONSTANTS["max_right_vergence"] + cfg.FISH_CONSTANTS["perception_field"]) * 180 / np.pi

    for time_points, angles in success_trajectories:
        
        time_array = np.array(time_points)
        angle_array = np.array(angles)
        convergence_mask = np.abs(angle_array - max_vergence) < convergence_threshold  # Within some degrees of convergence

        if np.any(convergence_mask):
            # Find first convergence time
            first_convergence_idx = np.where(convergence_mask)[0][0]
            convergence_time = time_array[first_convergence_idx]
            convergence_times.append(convergence_time)

    convergence_times = np.array(convergence_times)

    return convergence_times

# Calculate average vergence angles for each time step relative to outcome
def calculate_avg_speed_by_outcome(tracking_results):
    """Calculate average vergence angles by outcome and relative time step"""
    success_data = tracking_results[tracking_results['outcome'] == 'success']
    miss_data = tracking_results[tracking_results['outcome'] == 'miss']
    
    # Get maximum tracking duration to normalize time steps
    max_duration = tracking_results['tracking_duration'].max()
    
    success_trajectories = []
    miss_trajectories = []
    
    # Process success trajectories
    for _, row in success_data.iterrows():
        speeds = row['speeds']
        duration = row['tracking_duration']
        
        # Normalize to relative time (0 = start, 1 = end)
        if duration > 1:
            time_points = np.linspace(0, 1, len(speeds))
            success_trajectories.append((time_points, speeds))
    
    # Process miss trajectories
    for _, row in miss_data.iterrows():
        speeds = row['speeds']
        duration = row['tracking_duration']
        
        # Normalize to relative time (0 = start, 1 = end)
        if duration > 1:
            time_points = np.linspace(0, 1, len(speeds))
            miss_trajectories.append((time_points, speeds))

    return success_trajectories, miss_trajectories


def analyze_vergence_around_food_detection(dff, window_size=5):
    """
    Analyze vergence angle changes around food detection events.
    
    Parameters:
    - window_size: Number of time steps before and after detection to analyze
    """
    
    vergence_data = []
    
    # Group by agent and episode
    for (env_id, episode_index, agent_id), agent_data in dff.groupby(['env_id', 'episode_index', 'agent_id']):
        agent_data = agent_data.sort_values('time_step').reset_index(drop=True)
        
        # Find food detection events (when detected_food_ids changes from empty to non-empty)
        prev_detected = set()
        
        for i in range(len(agent_data)):
            current_row = agent_data.iloc[i]
            current_detected = set(current_row['detected_food_ids'])
            
            # Check if new food was detected
            newly_detected = current_detected - prev_detected
            
            if newly_detected and i >= window_size and i < len(agent_data) - window_size:
                # Get window around detection event
                window_data = agent_data.iloc[i - window_size:i + window_size + 1]
                
                # Extract vergence angles for this window
                for j, (_, row) in enumerate(window_data.iterrows()):
                    time_relative = j - window_size  # -window_size to +window_size
                    
                    vergence_data.append({
                        'env_id': env_id,
                        'episode_index': episode_index,
                        'agent_id': agent_id,
                        'detection_time_step': current_row['time_step'],
                        'time_relative': time_relative,
                        'vergence_angle_deg': row['vergence_angle_deg'],
                        'speed': row['speed'],
                        'eating_event': row['eating_event']
                    })
            
            prev_detected = current_detected
    
    return pd.DataFrame(vergence_data)

def analyze_vergence_speed_three_phases(tracking_results, dff, phase_duration=10):
    """
    Analyze vergence angle and speed in three normalized phases:
    1. Before detection (fixed duration)
    2. During tracking (detection to outcome - normalized)
    3. After outcome (fixed duration)
    """
    
    # Get the original dataframe with full time series
    phase_data = []
    
    # Group by individual tracking sequences
    for _, tracking_row in tracking_results.iterrows():
        env_id = tracking_row['env_id']
        episode_index = tracking_row['episode_index']
        agent_id = tracking_row['agent_id']
        start_time = tracking_row['start_time_step']
        duration = tracking_row['tracking_duration']
        outcome = tracking_row['outcome']
        
        # Get the full agent data for this episode
        agent_data = dff[
            (dff['env_id'] == env_id) & 
            (dff['episode_index'] == episode_index) & 
            (dff['agent_id'] == agent_id)
        ].sort_values('time_step').reset_index(drop=True)
        
        # Find the row indices corresponding to our time steps
        start_idx = agent_data[agent_data['time_step'] == start_time].index[0]
        end_idx = start_idx + duration
        
        # Phase 1: Before detection (fixed duration)
        if start_idx >= phase_duration:
            before_data = agent_data.iloc[start_idx - phase_duration:start_idx]
            before_vergence = before_data['vergence_angle_deg'].values
            before_speed = before_data['speed'].values
            before_time = np.linspace(-1, 0, len(before_vergence))  # -1 to 0
        else:
            before_vergence = []
            before_speed = []
            before_time = []
        
        # Phase 2: During tracking (detection to outcome - normalized)
        during_data = agent_data.iloc[start_idx:end_idx]
        during_vergence = during_data['vergence_angle_deg'].values
        during_speed = during_data['speed'].values
        during_time = np.linspace(0, 1, len(during_vergence))  # 0 to 1
        
        # Phase 3: After outcome (fixed duration)
        if end_idx + phase_duration <= len(agent_data):
            after_data = agent_data.iloc[end_idx:end_idx + phase_duration]
            after_vergence = after_data['vergence_angle_deg'].values
            after_speed = after_data['speed'].values
            after_time = np.linspace(1, 2, len(after_vergence))  # 1 to 2
        else:
            after_vergence = []
            after_speed = []
            after_time = []
        
        # Store the complete trajectory
        if len(before_vergence) > 0 and len(after_vergence) > 0:
            full_time = np.concatenate([before_time, during_time, after_time])
            full_vergence = np.concatenate([before_vergence, during_vergence, after_vergence])
            full_speed = np.concatenate([before_speed, during_speed, after_speed])
            
            phase_data.append({
                'outcome': outcome,
                'time': full_time,
                'vergence': full_vergence,
                'speed': full_speed,
                'duration': duration
            })
    
    return phase_data



########### TESTS ############
class TestSwimmingStatistics(unittest.TestCase):

    def test_calculate_polarization(self):
        # Test with aligned orientations
        orientations = [0, 0, 0, 0]
        self.assertAlmostEqual(calculate_polarization(orientations), 1.0, places=5)

        # Test with anti-aligned orientations
        orientations = [0, np.pi, 0, np.pi]
        self.assertAlmostEqual(calculate_polarization(orientations), -1.0, places=5)

        # Test with random orientations
        orientations = [0, np.pi / 2, np.pi, 3 * np.pi / 2]
        self.assertAlmostEqual(calculate_polarization(orientations), 0.0, places=5)

    def test_calculate_cohesion(self):
        # Test with two positions
        positions = [[0, 0], [3, 4]]  # Distance is 5
        self.assertAlmostEqual(calculate_cohesion(positions), 5.0, places=5)

        # Test with multiple positions
        positions = [[0, 0], [1, 1], [2, 2], [3, 3]]
        expected_cohesion = np.mean(
            [np.sqrt(2)] * 4
        )  # Each point has sqrt(2) as the nearest distance
        self.assertAlmostEqual(
            calculate_cohesion(positions), expected_cohesion, places=5
        )

        # Test with less than two positions
        positions = [[0, 0]]
        self.assertTrue(np.isnan(calculate_cohesion(positions)))


def bin_agent_size(dff, num_bins=3, bin_column='size_bin'):    
    edges = np.linspace(dff['agent_size'].min(),
                        dff['agent_size'].max(),
                        num=4)
    bin_labels = [f"{edges[i]:.2f}–{edges[i+1]:.2f}" for i in range(num_bins)]
    dff[bin_column] = pd.cut(dff['agent_size'],
                            bins=edges,
                            labels=bin_labels,
                            include_lowest=True)
    return dff, bin_labels


# ——— Bin agent_size into 3 equal‐width intervals ———
def plot_behavior_densities_1d(dff, bin_column='size_bin'):
    fig, axes = plt.subplots(1, 2, figsize=(5, 3), sharey=False)
    bin_labels = dff[bin_column].cat.categories
    for ax, col in zip(axes, ['move_forward', 'turn_angle']):
        for label in bin_labels:
            subset = dff.loc[dff[bin_column] == label, col]
            sns.kdeplot(
                subset,
                ax=ax,
                bw_adjust=1.0,
                fill=False,
                clip=(subset.min(), subset.max()),
                label=label
            )
        ax.set_xlabel(col.replace('_', ' ').title(), fontsize=12)
        ax.set_yticks([])  # no y‐axis ticks
        for spine in ['top', 'right', 'left']:
            ax.spines[spine].set_visible(False)
        ax.spines['bottom'].set_linewidth(0.8)
        if ax is axes[1]:
            ax.legend(title='agent_size bin', loc='upper right')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    unittest.main()
