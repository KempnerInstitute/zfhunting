import tqdm
import argparse
import numpy as np
import pandas as pd
from itertools import combinations
from collections import Counter
from scipy.stats import mannwhitneyu

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import rcParams

import seaborn as sns

import utils_report as ru
from utils_figstyle import set_nature_style
from utils_figsaving import _make_fig

set_nature_style()


def calculate_pairwise_polarization(orientations):
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
    N = orientations.size
    if N < 2:
        raise ValueError("Needs 2 agents or more to calculate pairwise polarization.")

    diffs = orientations[:, None] - orientations[None, :]
    cos_diffs = np.cos(diffs)

    # We exclude diagonal (self-self) and duplicate pairs (use i<j)
    i, j = np.triu_indices(N, k=1)
    pairwise_cos = cos_diffs[i, j]

    return np.mean(pairwise_cos)


def classify_interaction(pos_a1, pos_a2, ori_a1, ori_a2, theta_close=np.pi/6, theta_opposite=5*np.pi/6):
    delta_pos = pos_a2 - pos_a1
    bearing_1_to_2 = np.arctan2(delta_pos[1], delta_pos[0])
    bearing_2_to_1 = np.arctan2(-delta_pos[1], -delta_pos[0])

    rel_angle1 = np.arctan2(np.sin(ori_a1 - bearing_1_to_2), np.cos(ori_a1 - bearing_1_to_2))
    rel_angle2 = np.arctan2(np.sin(ori_a2 - bearing_2_to_1), np.cos(ori_a2 - bearing_2_to_1))

    if np.abs(rel_angle1) < theta_close and np.abs(rel_angle2) < theta_close:
        return 'confronting'
    elif np.abs(rel_angle1) < theta_close and np.abs(rel_angle2) > theta_opposite:
        return 'chasing' # A1 is chasing A2
    elif np.abs(rel_angle2) < theta_close and np.abs(rel_angle1) > theta_opposite:
        return 'fleeing' # A2 is chasing A1
    elif np.abs(rel_angle1) > theta_opposite and np.abs(rel_angle2) > theta_opposite:
        return 'dispersing'
    else:
        return 'unaligned'


# def get_interactions_df_2fish(dff, contact_distance=2, 
#                               theta_close=np.pi/6, 
#                               theta_opposite=5*np.pi/6, 
#                               reduced_threshold_distance=5):
#     num_agents = dff["agent_id"].nunique()
#     if num_agents != 2:
#         raise ValueError(f"Expected 2 agents, but found {num_agents} agents in the DataFrame.")

#     # Ensure proper sorting for time continuity
#     dff = dff.sort_values(by=["env_id", "episode_index", "time_step", "agent_id"])

#     interaction_df_records = []
#     timestep_df_records = []
#     for (env_id, episode_index), group in tqdm.tqdm(dff.groupby(["env_id", "episode_index"])):
#         agents = group["agent_id"].unique()
#         if len(agents) != 2:
#             continue

#         # Extract agent dataframes
#         a1, a2 = agents
#         df_a1 = group[group["agent_id"] == a1].reset_index(drop=True)
#         df_a2 = group[group["agent_id"] == a2].reset_index(drop=True)

#         # Ensure aligned by time_step
#         assert np.all(df_a1["time_step"].values == df_a2["time_step"].values)
#         time_steps = df_a1["time_step"].values

#         # Compute mutual has_nearby
#         mutual_nearby = df_a1["has_nearby"].values & df_a2["has_nearby"].values

#         # Identify contiguous interaction segments
#         interaction_segments = []
#         in_interaction = False
#         for i, val in enumerate(mutual_nearby):
#             if val and not in_interaction:
#                 # Start of interaction
#                 start_idx = i
#                 in_interaction = True
#             elif not val and in_interaction:
#                 # End of interaction
#                 end_idx = i - 1
#                 interaction_segments.append((start_idx, end_idx))
#                 in_interaction = False
#         # Handle case where interaction extends to end
#         if in_interaction:
#             end_idx = len(mutual_nearby) - 1
#             interaction_segments.append((start_idx, end_idx))

#         # Build interaction records from the identified segments
#         last_end_time = None
#         for interaction_index, (start_idx, end_idx) in enumerate(interaction_segments):
#             t0 = time_steps[start_idx]
#             t1 = time_steps[end_idx]

#             time_since_last = (
#                 t0 - last_end_time if last_end_time is not None else np.nan
#             )
#             last_end_time = t1

#             # Subset rows for this interaction
#             df_seg_a1 = df_a1.iloc[start_idx:end_idx+1]
#             df_seg_a2 = df_a2.iloc[start_idx:end_idx+1]

#             # Compute stats
#             was_bitten_either = df_seg_a1["was_bitten"].values | df_seg_a2["was_bitten"].values
#             food_observed_either = df_seg_a1["food_observed"].values | df_seg_a2["food_observed"].values
#             has_eating_either = df_seg_a1["eating_event"].any() or df_seg_a2["eating_event"].any()
#             bite_other_fish_either = df_seg_a1["bite_other_fish"].values | df_seg_a2["bite_other_fish"].values

#             distances = df_seg_a1["distance_to_nearest_agent"].values
#             has_contact = np.any(distances <= contact_distance)

#             polarizations = []
#             interaction_classes = []
#             for i in range(len(df_seg_a1)):
#                 ori1 = df_seg_a1.iloc[i]["orientation"]
#                 ori2 = df_seg_a2.iloc[i]["orientation"]
#                 pos1 = np.array(df_seg_a1.iloc[i]["position"])
#                 pos2 = np.array(df_seg_a2.iloc[i]["position"])

#                 polarizations.append(calculate_pairwise_polarization([ori1, ori2]))
#                 interaction_class = classify_interaction(
#                     pos1, pos2, ori1, ori2,
#                     theta_close=theta_close,
#                     theta_opposite=theta_opposite
#                 )
#                 interaction_classes.append(interaction_class)

#             # Determine dominant interaction class (mode)
#             dominant_interaction_class = Counter(interaction_classes).most_common(1)[0][0]

#             # Filter interaction classes based on distance threshold
#             filtered_classes = pd.Series(interaction_classes)[distances <= reduced_threshold_distance]
#             # print(f"Filtered classes: {filtered_classes}", distances)
#             if filtered_classes.empty:
#                 dominant_interaction_class_filtered = 'unaligned'
#             else:
#                 dominant_interaction_class_filtered = Counter(filtered_classes.to_list()).most_common(1)[0][0]

#             interaction_record = {
#                 "env_id": env_id,
#                 "episode_index": episode_index,
#                 "interaction_index": interaction_index,
#                 "agents_involved": (a1, a2),
#                 "agent_size_a1": df_seg_a1["agent_size"].iloc[0],
#                 "agent_size_a2": df_seg_a2["agent_size"].iloc[0],
#                 "start_time_step": t0,
#                 "end_time_step": t1,
#                 "time_since_last_interaction": time_since_last,
#                 "has_biting_either": np.any(was_bitten_either),
#                 "has_contact": has_contact,
#                 "food_observed_either": np.any(food_observed_either),
#                 "has_eating_either": has_eating_either,
#                 "mean_distance": np.mean(distances),
#                 "max_distance": np.max(distances),
#                 "min_distance": np.min(distances),
#                 "mean_polarization": np.mean(polarizations),
#                 "dominant_interaction_class": dominant_interaction_class,
#                 "dominant_interaction_class_filtered": dominant_interaction_class_filtered,
#                 "reduced_threshold_distance": reduced_threshold_distance,
#                 "contact_distance": contact_distance,
#                 # "has_distance_beyond_10cm": np.any(distances > 10), # DEBUG
#             }
#             interaction_df_records.append(interaction_record)

#             timestep_record = pd.DataFrame({
#                 "env_id": env_id,
#                 "episode_index": episode_index,
#                 "time_step": range(t0, t1 + 1),
#                 "focal_agent": a1,
#                 "other_agent": a2,
#                 "interaction_index": interaction_index,
#                 "distance": distances,
#                 "interaction_class": interaction_classes,
#                 "polarization_pairwise": polarizations,
#                 "position_a1": df_seg_a1["position"].tolist(),
#                 "orientation_a1": df_seg_a1["orientation"].tolist(),
#                 "position_a2": df_seg_a2["position"].tolist(),
#                 "orientation_a2": df_seg_a2["orientation"].tolist(),
#                 "was_bitten_a1": df_seg_a1["was_bitten"].values,
#                 "bite_other_fish_a1": df_seg_a1["bite_other_fish"].values,
#                 "food_observed_a1": df_seg_a1["food_observed"].values,
#                 "was_bitten_a2": df_seg_a2["was_bitten"].values,
#                 "bite_other_fish_a2": df_seg_a2["bite_other_fish"].values,
#                 "food_observed_a2": df_seg_a2["food_observed"].values,
#             })
#             timestep_df_records.append(timestep_record)

#     interactions_df = pd.DataFrame(interaction_df_records)
#     timestep_df = pd.concat(timestep_df_records, ignore_index=True)
#     return interactions_df, timestep_df

def get_interactions_df_Nfish(dff, contact_distance=2, 
                              theta_close=np.pi/6, 
                              theta_opposite=5*np.pi/6, 
                              reduced_threshold_distance=5):
    dff = dff.sort_values(by=["env_id", "episode_index", "time_step", "agent_id"])

    interaction_df_records = []
    timestep_df_records = []
    print(f"Processing <env_id,episode_index> pairs from {len(dff)} rows for interactions...")
    for (env_id, episode_index), group in tqdm.tqdm(dff.groupby(["env_id", "episode_index"])):
        agents = sorted(group["agent_id"].unique())
        for a1, a2 in combinations(agents, 2):
            df_a1 = group[group["agent_id"] == a1].reset_index(drop=True)
            df_a2 = group[group["agent_id"] == a2].reset_index(drop=True)

            # Ensure aligned by time_step
            if len(df_a1) != len(df_a2) or not np.all(df_a1["time_step"].values == df_a2["time_step"].values):
                print(f"[WARN] Unaligned data for env_id={env_id}, episode_index={episode_index}, agents=({a1}, {a2})")
                continue  # skip unaligned data

            time_steps = df_a1["time_step"].values
            mutual_nearby = df_a1["has_nearby"].values & df_a2["has_nearby"].values

            # Identify contiguous interaction segments
            interaction_segments = []
            in_interaction = False
            for i, val in enumerate(mutual_nearby):
                if val and not in_interaction:
                    # Start of interaction
                    start_idx = i
                    in_interaction = True
                elif not val and in_interaction:
                    # End of interaction
                    end_idx = i - 1
                    interaction_segments.append((start_idx, end_idx))
                    in_interaction = False
            # Handle case where interaction extends to end
            if in_interaction:
                end_idx = len(mutual_nearby) - 1
                interaction_segments.append((start_idx, end_idx))

            # Build interaction records from the identified segments
            last_end_time = None
            for interaction_index, (start_idx, end_idx) in enumerate(interaction_segments):
                t0 = time_steps[start_idx]
                t1 = time_steps[end_idx]

                time_since_last = (
                    t0 - last_end_time if last_end_time is not None else np.nan
                )
                last_end_time = t1

                # Subset rows for this interaction
                df_seg_a1 = df_a1.iloc[start_idx:end_idx+1]
                df_seg_a2 = df_a2.iloc[start_idx:end_idx+1]

                if len(df_seg_a1) == 0 or len(df_seg_a2) == 0:
                    continue
                if len(df_seg_a1) != len(df_seg_a2):
                    print(f"[WARN] Mismatched segment lengths for env_id={env_id}, episode_index={episode_index}, agents=({a1}, {a2})")
                    continue

                # Compute stats
                was_bitten_either = df_seg_a1["was_bitten"].values | df_seg_a2["was_bitten"].values
                food_observed_either = df_seg_a1["food_observed"].values | df_seg_a2["food_observed"].values
                has_eating_either = df_seg_a1["eating_event"].any() or df_seg_a2["eating_event"].any()
                bite_other_fish_either = df_seg_a1["bite_other_fish"].values | df_seg_a2["bite_other_fish"].values

                distances = df_seg_a1["distance_to_nearest_agent"].values
                has_contact = np.any(distances <= contact_distance)

                polarizations = []
                interaction_classes = []
                for i in range(len(df_seg_a1)):
                    ori1 = df_seg_a1.iloc[i]["orientation"]
                    ori2 = df_seg_a2.iloc[i]["orientation"]
                    pos1 = np.array(df_seg_a1.iloc[i]["position"])
                    pos2 = np.array(df_seg_a2.iloc[i]["position"])

                    polarizations.append(calculate_pairwise_polarization([ori1, ori2]))
                    interaction_class = classify_interaction(
                        pos1, pos2, ori1, ori2,
                        theta_close=theta_close,
                        theta_opposite=theta_opposite
                    )
                    interaction_classes.append(interaction_class)

                # Determine dominant interaction class (mode)
                dominant_interaction_class = Counter(interaction_classes).most_common(1)[0][0]

                # Filter interaction classes based on distance threshold
                filtered_classes = pd.Series(interaction_classes)[distances <= reduced_threshold_distance]
                # print(f"Filtered classes: {filtered_classes}", distances)
                if filtered_classes.empty:
                    dominant_interaction_class_filtered = 'unaligned'
                else:
                    dominant_interaction_class_filtered = Counter(filtered_classes.to_list()).most_common(1)[0][0]

                interaction_record = {
                    "env_id": env_id,
                    "episode_index": episode_index,
                    "interaction_index": interaction_index,
                    "focal_agent": a1,
                    "other_agent": a2,
                    "agent_size_a1": df_seg_a1["agent_size"].iloc[0],
                    "agent_size_a2": df_seg_a2["agent_size"].iloc[0],
                    "start_time_step": t0,
                    "end_time_step": t1,
                    "time_since_last_interaction": time_since_last,
                    "has_biting_either": np.any(was_bitten_either),
                    "has_contact": has_contact,
                    "food_observed_either": np.any(food_observed_either),
                    "has_eating_either": has_eating_either,
                    "mean_distance": np.mean(distances),
                    "max_distance": np.max(distances),
                    "min_distance": np.min(distances),
                    "mean_polarization": np.mean(polarizations),
                    "dominant_interaction_class": dominant_interaction_class,
                    "dominant_interaction_class_filtered": dominant_interaction_class_filtered,
                    "reduced_threshold_distance": reduced_threshold_distance,
                    "contact_distance": contact_distance,
                    # "has_distance_beyond_10cm": np.any(distances > 10), # DEBUG
                }
                interaction_df_records.append(interaction_record)

                timestep_record = pd.DataFrame({
                    "env_id": env_id,
                    "episode_index": episode_index,
                    "time_step": range(t0, t1 + 1),
                    "focal_agent": a1,
                    "other_agent": a2,
                    "interaction_index": interaction_index, # to merge with interaction_df
                    "distance": distances,
                    "interaction_class": interaction_classes,
                    "polarization_pairwise": polarizations,
                    "position_a1": df_seg_a1["position"].tolist(),
                    "orientation_a1": df_seg_a1["orientation"].tolist(),
                    "position_a2": df_seg_a2["position"].tolist(),
                    "orientation_a2": df_seg_a2["orientation"].tolist(),
                    "was_bitten_a1": df_seg_a1["was_bitten"].values,
                    "bite_other_fish_a1": df_seg_a1["bite_other_fish"].values,
                    "food_observed_a1": df_seg_a1["food_observed"].values,
                    "was_bitten_a2": df_seg_a2["was_bitten"].values,
                    "bite_other_fish_a2": df_seg_a2["bite_other_fish"].values,
                    "food_observed_a2": df_seg_a2["food_observed"].values,
                })
                timestep_df_records.append(timestep_record)

    interactions_df = pd.DataFrame(interaction_df_records)
    timestep_df = pd.concat(timestep_df_records, ignore_index=True)
    return interactions_df, timestep_df


def merge_pairwise_interaction_timestep_df_with_dff(dff, timestep_df):
    """
    Merge per-agent interaction features from timestep_df back into dff,
    accounting for symmetry in (focal_agent, other_agent) combinations.
    """

    # Columns to keep from timestep_df
    columns_to_keep = [
        'env_id', 'episode_index', 'time_step',
        'focal_agent', 'other_agent',
        'interaction_index',  # optional, good for filtering later
        'interaction_class', 'polarization_pairwise'
    ]

    timestep_df_trimmed = timestep_df[columns_to_keep].copy()
    timestep_df_swapped = timestep_df_trimmed.rename(columns={
        'focal_agent': 'other_agent',
        'other_agent': 'focal_agent'
    })
    timestep_df_all = pd.concat([timestep_df_trimmed, timestep_df_swapped], ignore_index=True)

    # Check for duplicates after concatenation (this affects merging; shows up when n_agents > 2)
    num_duplicates = timestep_df_all.duplicated(subset=['env_id', 'episode_index', 'time_step', 'focal_agent']).sum()
    print(f"Number of duplicate rows after concatenation: {num_duplicates}, timestep_df.shape: {timestep_df.shape}")
    if num_duplicates > 0:
        num_agents = set(timestep_df['focal_agent'].unique()).union(set(timestep_df['other_agent'].unique()))
        print(f"Number of unique agents in timestep_df: {num_agents} [Problematic when > 2]")

        timestep_df_all = timestep_df_all.sort_values(by=['env_id', 'episode_index', 'time_step', 'focal_agent'])
        timestep_df_all = timestep_df_all.drop_duplicates(subset=['env_id', 'episode_index', 'time_step', 'focal_agent'])
        print("[WARN] Duplicates found, dropped rows to ensure unique focal_agent per timestep.")

    # Merge
    timestep_df_all = timestep_df_all.rename(columns={'focal_agent': 'agent_id'})
    merged_df = pd.merge(
        dff,
        timestep_df_all,
        how='left',
        on=['env_id', 'episode_index', 'time_step', 'agent_id']
    )
    return merged_df


def plot_agent_size_by_role(
        interactions_df,
        filter_type=["dominant_interaction_class", "dominant_interaction_class_filtered"][1],
        outputs_folder=None,
        outfile_base=None,
):
    """
    Plot agent size by role (chaser vs chased) using boxplot.
    """
    # Dominant interaction class
    
    chasing_df = interactions_df[interactions_df[filter_type] == 'chasing']
    fleeing_df = interactions_df[interactions_df[filter_type] == 'fleeing']

    chaser_sizes = pd.concat([
        chasing_df['agent_size_a1'],
        fleeing_df['agent_size_a2']
    ])

    chased_sizes = pd.concat([
        chasing_df['agent_size_a2'],
        fleeing_df['agent_size_a1']
    ])

    # Mann-Whitney U Test
    stat, pval = mannwhitneyu(chaser_sizes, chased_sizes, alternative='two-sided')

    # Boxplot
    data_q1 = pd.DataFrame({
        'agent_size': pd.concat([chaser_sizes, chased_sizes]),
        'role': ['chaser'] * len(chaser_sizes) + ['chased'] * len(chased_sizes)
    })

    fig, ax = _make_fig(1, 1)
    sns.boxplot(x='role', y='agent_size', data=data_q1, ax=ax)
    # ax.set_title(f"Agent size by role (chaser vs chased)\nMann-Whitney U p={pval:.4f}")
    ax.set_title(f"Mann-Whitney U p={pval:.4f}")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    if outfile_base is not None:
        fname = f"{outfile_base}_agent_size_by_role_boxplot.png"
    elif outputs_folder:
        fname = f"{outputs_folder}/agent_size_by_role_boxplot.png"
    else:
        print("[WARN] No outputs_folder or outfile_base provided, not saving agent_size_by_role boxplot.")
        plt.show()
        return

    plt.tight_layout()
    fig.savefig(fname, bbox_inches='tight', dpi=300)
    print(f"Saved agent size by role plot to {fname}")

    # 2D Histogram
    fig, ax = _make_fig(1, 1)
    h = ax.hist2d(chaser_sizes, chased_sizes, bins=5, cmap='Blues', cmin=1)
    ax.set_xlabel("Chaser agent size")
    ax.set_ylabel("Chased agent size")
    fig.colorbar(h[3], ax=ax, label='Counts')

    if outfile_base is not None:
        fname = f"{outfile_base}_agent_size_by_role_hist2d.png"
    elif outputs_folder:
        fname = f"{outputs_folder}/agent_size_by_role_hist2d.png"
    else:
        print("[WARN] No outputs_folder or outfile_base provided, not saving agent_size_by_role histogram.")
        plt.show()
        return

    plt.tight_layout()
    fig.savefig(fname, bbox_inches='tight', dpi=300)
    print(f"Saved agent size by role plot to {fname}")


def plot_agent_size_in_confrontations_1d(
    interactions_df,
    filter_type=["dominant_interaction_class", "dominant_interaction_class_filtered"][1],
    outputs_folder=None,
    outfile_base=None
):
    """
    Plot agent size in confrontations using histogram.
    """
    confronting_df = interactions_df[interactions_df[filter_type] == 'confronting']
    confront_sizes = pd.concat([
        confronting_df['agent_size_a1'],
        confronting_df['agent_size_a2']
    ])

    print(f"Mean agent size in confrontations: {confront_sizes.mean():.3f}")
    print(f"Median agent size in confrontations: {confront_sizes.median():.3f}")

    plt.figure(figsize=(6, 4))
    sns.histplot(confront_sizes, bins=10, kde=True)
    plt.title("Agent size distribution in confrontations")
    plt.xlabel("agent_size")

    if outfile_base is not None:
        fname = f"{outfile_base}_plot_agent_size_in_confrontations_1d.png"
    elif outputs_folder:
        fname = f"{outputs_folder}/plot_agent_size_in_confrontations_1d.png"
    else:
        print("[WARN] No outputs_folder or outfile_base provided, not saving agent_size_in_confrontations_1d plot.")
        plt.show()
        return

    plt.tight_layout()
    plt.savefig(fname, bbox_inches='tight', dpi=300)
    print(f"Saved agent size in confrontations plot to {fname}")


def plot_agent_size_in_confrontations_2d(
    interactions_df,
    filter_type=["dominant_interaction_class", "dominant_interaction_class_filtered"][1],
    outputs_folder=None,
    outfile_base=None
):
    """
    Plot agent size in confrontations using histogram.
    """
    confronting_df = interactions_df[interactions_df[filter_type] == 'confronting']

    # Canonicalize agent size pairs (unordered)
    ordered_sizes = np.sort(confronting_df[["agent_size_a1", "agent_size_a2"]].values, axis=1)
    size_min = ordered_sizes[:, 0]
    size_max = ordered_sizes[:, 1]

    # Plot unique unordered pairs
    fig, ax = _make_fig(1, 1)
    h = ax.hist2d(size_min, size_max, bins=5, cmap='Blues', cmin=1)
    ax.set_xlabel("Smaller agent size")
    ax.set_ylabel("Larger agent size")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Unique (unordered) agent size pairs in confrontations")
    fig.colorbar(h[3], ax=ax, label='Counts')
    plt.tight_layout()

    if outfile_base is not None:
        fname = f"{outfile_base}_plot_agent_size_in_confrontations_unique_pairs.png"
    elif outputs_folder:
        fname = f"{outputs_folder}/plot_agent_size_in_confrontations_unique_pairs.png"
    else:
        print("[WARN] No outputs_folder or outfile_base provided, not saving confrontation size plot.")
        plt.show()
        return

    plt.savefig(fname, bbox_inches='tight', dpi=300)
    print(f"Saved confrontation size plot to {fname}")


def plot_interaction_classes_by_distance_from_timestep_df(timestep_df, bin_size_cm=1, normalize=False, outputs_folder=None, outfile_base=None):
    """
    Visualize the distribution of interaction classes as a stacked barplot
    binned by distance (1 cm bins by default).

    Parameters:
    -----------
    timestep_df : pd.DataFrame
        The timestep-level DataFrame returned by get_interactions_df_2fish, which includes
        `distance` and `interaction_class` columns.

    bin_size_cm : float
        Bin size for distance in cm (default = 1 cm).

    normalize : bool
        Whether to normalize bar heights to sum to 1 per bin (i.e., plot proportions).
    """
    # Drop missing distance values (if any)
    df = timestep_df[["distance", "interaction_class"]].dropna()

    # Bin distances
    df['distance_bin'] = (df['distance'] // bin_size_cm) * bin_size_cm

    # Count occurrences
    count_table = df.groupby(['distance_bin', 'interaction_class']).size().unstack(fill_value=0)

    # Sort bins for proper x-axis order
    count_table = count_table.sort_index()

    # Normalize if requested
    if normalize:
        count_table = count_table.div(count_table.sum(axis=1), axis=0)

    # Plot
    fig, ax = _make_fig(1, 1, width_multiplier=2)
    count_table.plot(
        kind='bar',
        stacked=True,
        ax=ax,
        colormap='tab20',
        width=1.0
    )
    ax.set_xlabel("Distance to other agent (cm)")
    ylabel = "Proportion" if normalize else "Count"
    ax.set_ylabel(ylabel)
    # ax.set_title("Interaction class distribution vs distance")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='x', rotation=90)
    plt.tight_layout()

    if outfile_base is not None:
        fname = f"{outfile_base}_interaction_classes_by_distance.png"
    elif outputs_folder:
        fname = f"{outputs_folder}/interaction_classes_by_distance.png"
    else:
        print("[WARN] No outputs_folder or outfile_base provided, not saving interaction classes by distance plot.")
        plt.show()
        return

    plt.savefig(fname, bbox_inches='tight', dpi=300)
    print(f"Saved interaction classes by distance plot to {fname}")



def plot_interaction_with_metadata(interaction_record, timestep_df, outputs_folder=None, outfile_base=None):
    interaction_index = interaction_record["interaction_index"]
    env_id = interaction_record["env_id"]
    episode_index = interaction_record["episode_index"]

    df = timestep_df[
        (timestep_df["env_id"] == env_id) &
        (timestep_df["episode_index"] == episode_index) &
        (timestep_df["interaction_index"] == interaction_index)
    ]

    agent_id_a1 = df["focal_agent"].iloc[0]
    agent_id_a2 = df["other_agent"].iloc[0]
    pos_a1 = np.array(df["position_a1"].tolist())
    pos_a2 = np.array(df["position_a2"].tolist())
    ori_a1 = np.array(df["orientation_a1"].tolist())
    ori_a2 = np.array(df["orientation_a2"].tolist())
    distances = df["distance"].values
    polarizations = df["polarization_pairwise"].values
    was_bitten_a1 = df["was_bitten_a1"].values
    was_bitten_a2 = df["was_bitten_a2"].values
    time = np.arange(len(df))

# Time-normalized colormaps
    norm = mcolors.Normalize(vmin=0, vmax=len(df) - 1)
    cmap_a1 = cm.get_cmap("Blues")
    cmap_a2 = cm.get_cmap("Greens")
    colors_a1 = cmap_a1(norm(time))
    colors_a2 = cmap_a2(norm(time))

    fig = plt.figure(figsize=(8, 9))
    gs = fig.add_gridspec(5, 1, height_ratios=[4, 0.6, 1, 1, 0.2], hspace=0.0)
    ax_traj = fig.add_subplot(gs[0])
    spacer_ax = fig.add_subplot(gs[1])
    ax_dist = fig.add_subplot(gs[2])
    ax_pol = fig.add_subplot(gs[3], sharex=ax_dist)
    cax = fig.add_subplot(gs[4])
    spacer_ax.axis('off')

    # Quiver for Agent A1
    U1 = 0.2 * np.cos(ori_a1)
    V1 = 0.2 * np.sin(ori_a1)
    ax_traj.quiver(
        pos_a1[:, 0], pos_a1[:, 1], U1, V1,
        angles='xy', scale_units='xy', scale=1,
        color=colors_a1, alpha=0.8, width=0.009, label=f'Agent {agent_id_a1}'
    )

    # Quiver for Agent A2
    U2 = 0.2 * np.cos(ori_a2)
    V2 = 0.2 * np.sin(ori_a2)
    ax_traj.quiver(
        pos_a2[:, 0], pos_a2[:, 1], U2, V2,
        angles='xy', scale_units='xy', scale=1,
        color=colors_a2, alpha=0.8, width=0.009, label=f'Agent {agent_id_a2}'
    )

    # Mark times when each agent was bitten
    bitten_indices_a1 = np.where(was_bitten_a1)[0]
    bitten_indices_a2 = np.where(was_bitten_a2)[0]

    ax_traj.scatter(
        pos_a1[bitten_indices_a1, 0], pos_a1[bitten_indices_a1, 1],
        color='red', marker='o', s=50, label=f'A{agent_id_a1} Bitten'
    )
    ax_traj.scatter(
        pos_a2[bitten_indices_a2, 0], pos_a2[bitten_indices_a2, 1],
        color='red', marker='x', s=50, label=f'A{agent_id_a2} Bitten'
    )

    ax_traj.set_ylabel("Y position")
    ax_traj.set_xlabel("X position")
    ax_traj.set_title(f"Interaction {interaction_index} Δt={interaction_record['end_time_step'] - interaction_record['start_time_step'] + 1}")
    ax_traj.set_aspect('equal')

    # Legend using final color from each agent's cmap
    legend_elements = [
        Line2D([0], [0], color=cmap_a1(0.75), lw=2, label=f'Agent {agent_id_a1}'),
        Line2D([0], [0], color=cmap_a2(0.75), lw=2, label=f'Agent {agent_id_a2}'),
        Line2D([0], [0], marker='o', color='red', label=f'{agent_id_a1} Bitten',
               markersize=6, linestyle='None'),
        Line2D([0], [0], marker='x', color='red', label=f'{agent_id_a2} Bitten',
               markersize=6, linestyle='None')
    ]
    ax_traj.legend(handles=legend_elements, loc='upper right', fontsize=8)

    sm = cm.ScalarMappable(cmap=cmap_a1, norm=norm)
    fig.colorbar(sm, cax=cax, orientation='horizontal', label='Timestep')

    # Distance plot
    ax_dist.plot(time, distances, color='purple')
    ax_dist.set_ylabel("Distance")
    ax_dist.set_ylim(bottom=0)  # Ensure min is 0
    ax_dist.tick_params(labelbottom=False)
    ax_dist.set_xlim(0, len(df) - 1)

    # Polarization plot
    ax_pol.plot(time, polarizations, color='green')
    ax_pol.set_ylabel("Polarization")
    ax_pol.set_ylim(-1.05, 1.05)
    ax_pol.set_xticklabels([])

    if outfile_base is not None:
        fname = f"{outfile_base}_interaction_{interaction_index}_env{env_id}_ep{episode_index}.png"
    elif outputs_folder:
        fname = f"{outputs_folder}/interaction_{interaction_index}_env{env_id}_ep{episode_index}.png"
    else:
        print("[WARN] No outputs_folder or outfile_base provided, not saving interaction plot.")
        plt.show()
        return fig

    plt.tight_layout()
    plt.savefig(fname, bbox_inches='tight', dpi=300)
    print(f"Saved interaction plot to {fname}")

    return fig

def plot_polarization_histogram(timestep_df, outputs_folder=None, outfile_base=None):
    all_polarizations = timestep_df["polarization_pairwise"].dropna().values
    fig, ax = _make_fig(1, 1)
    ax.hist(all_polarizations, bins=30, range=(-1, 1), density=True, color='C0', edgecolor='black')
    ax.set_xlabel("Polarization")
    ax.set_ylabel("Density")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


    if outfile_base is not None:
        fname = f"{outfile_base}_polarization_histogram.png"
    elif outputs_folder:
        fname = f"{outputs_folder}/polarization_histogram.png"
    else:
        print("[WARN] No outputs_folder or outfile_base provided, not saving polarization histogram.")
        plt.show()
        return

    plt.tight_layout()
    plt.savefig(fname, bbox_inches='tight', dpi=300)
    print(f"Saved polarization histogram to {fname}")


def plot_agent_size_histogram(interactions_df, outputs_folder=None, outfile_base=None):
    fig, ax = _make_fig(1, 1)
    interactions_df["max_distance"].hist(bins=50, color='C0', alpha=1, ax=ax, grid=False)
    ax.set_xlabel("Max Distance (cm)")
    ax.set_ylabel("Frequency")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    if outfile_base is not None:
        fname = f"{outfile_base}_agent_size_histogram.png"
    elif outputs_folder:
        fname = f"{outputs_folder}/agent_size_histogram.png"
    else:
        print("[WARN] No outputs_folder or outfile_base provided, not saving agent size histogram.")
        plt.show()
        return

    plt.tight_layout()
    plt.savefig(fname, bbox_inches='tight', dpi=300)
    print(f"Saved agent size histogram to {fname}")


def run_pairwise_report(dff, outfile_base):
    interactions_df, timestep_df = get_interactions_df_Nfish(dff, theta_close=np.pi/4, theta_opposite=3*np.pi/4)
    print(interactions_df["dominant_interaction_class"].value_counts())
    print(interactions_df["dominant_interaction_class_filtered"].value_counts())

    # PLOTS
    plot_polarization_histogram(timestep_df, outfile_base=outfile_base)

    plot_agent_size_histogram(interactions_df, outfile_base=outfile_base)

    plot_agent_size_by_role(
        interactions_df,
        filter_type=["dominant_interaction_class", "dominant_interaction_class_filtered"][1],
        outfile_base=outfile_base
    )
    
    plot_agent_size_in_confrontations_2d(
        interactions_df,
        filter_type=["dominant_interaction_class", "dominant_interaction_class_filtered"][1],
        outfile_base=outfile_base
    )

    # For visualization, collapse the "chasing" and "fleeing" labels into "chasing_fleeing" in the "interaction_class" column
    timestep_df_alt = timestep_df.copy()
    timestep_df_alt["interaction_class"] = timestep_df_alt["interaction_class"].replace(
        {"chasing": "chasing_fleeing", "fleeing": "chasing_fleeing"}
    )
    plot_interaction_classes_by_distance_from_timestep_df(timestep_df_alt, bin_size_cm=1, normalize=False, outfile_base=outfile_base)
    # plot_interaction_classes_by_distance_from_timestep_df(timestep_df, bin_size_cm=1, normalize=False, outputs_folder=outputs_folder)

    subset = interactions_df.query("dominant_interaction_class == 'confronting'").head(3)
    for _, record in subset.iterrows():
        fig = plot_interaction_with_metadata(record, timestep_df, outfile_base=outfile_base)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--flat_pkl_file", type=str, help="Path to input pickle file")
    parser.add_argument(
        "--model_seed", type=int, default=137, help="Random seed for classifier"
    )
    parser.add_argument(
        "--name_mode",
        type=str,
        default="name",
        choices=["name", "short_name"],
        help="Feature label mode",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="foraging",
        help="Task e.g., 'foraging', '2fip', etc.",
    )
    args = parser.parse_args()

    flat_pkl_file = args.flat_pkl_file
    model_seed = args.model_seed
    task = args.task
    name_mode = args.name_mode

    dff, outputs_folder, pkl_str = ru.load_flat_pkl_file(flat_pkl_file, task=task)
    dff.columns

    num_agents = dff["agent_id"].nunique()
    if ("2f" in task) and (num_agents != 2):
        raise ValueError(f"Expected 2 agents, but found {num_agents} agents in the DataFrame. Task: {task}")
    

    interactions_df, timestep_df = get_interactions_df_Nfish(dff, theta_close=np.pi/4, theta_opposite=3*np.pi/4)
    print(interactions_df["dominant_interaction_class"].value_counts())
    print(interactions_df["dominant_interaction_class_filtered"].value_counts())

    # PLOTS
    plot_polarization_histogram(timestep_df, outputs_folder=outputs_folder)

    plot_agent_size_histogram(interactions_df, outputs_folder=outputs_folder)

    plot_agent_size_by_role(
        interactions_df,
        filter_type=["dominant_interaction_class", "dominant_interaction_class_filtered"][1],
        outputs_folder=outputs_folder
    )
    
    plot_agent_size_in_confrontations_2d(
        interactions_df,
        filter_type=["dominant_interaction_class", "dominant_interaction_class_filtered"][1],
        outputs_folder=outputs_folder
    )
    

    # For visualization, collapse the "chasing" and "fleeing" labels into "chasing_fleeing" in the "interaction_class" column
    timestep_df_alt = timestep_df.copy()
    timestep_df_alt["interaction_class"] = timestep_df_alt["interaction_class"].replace(
        {"chasing": "chasing_fleeing", "fleeing": "chasing_fleeing"}
    )
    plot_interaction_classes_by_distance_from_timestep_df(timestep_df_alt, bin_size_cm=1, normalize=False)
    # plot_interaction_classes_by_distance_from_timestep_df(timestep_df, bin_size_cm=1, normalize=False, outputs_folder=outputs_folder)


    subset = interactions_df.query("dominant_interaction_class == 'confronting'").head(3)
    for _, record in subset.iterrows():
        fig = plot_interaction_with_metadata(record, timestep_df, outputs_folder=outputs_folder)
        plt.show()