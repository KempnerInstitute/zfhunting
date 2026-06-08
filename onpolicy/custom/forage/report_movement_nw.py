#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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
    # Parses the command line arguments below


def get_latest_flat_pkl_file(input_dir="./"):
    pkl_files = glob.glob(input_dir + "/*.pkl")
    pkl_files = [f for f in pkl_files if "flat" in f]
    if not pkl_files:
        raise FileNotFoundError("No .pkl files found in the current directory.")
    latest_pkl_file = max(pkl_files, key=os.path.getctime)
    return latest_pkl_file

default_dir = "./forage/results/rmappo-MultiAgentForagingEnv-check/"

default_dir = "/srv/marl/nathanwu/rmappo-MultiAgentForagingEnv-check/20250921_174035_1_bao_vd_-0.006_lmp_-0.01_ltp_-0.01_fdr_10_run_1/outputs"
default_dir = "/srv/marl/nathanwu/rmappo-MultiAgentForagingEnv-check/20250922_213551_1_bao_vd_-0.006_lmp_-0.01_ltp_-0.01_fdr_10_run_6/outputs"

#seed 0
default_dir = "/srv/marl/nathanwu/rmappo-MultiAgentForagingEnv-check/20250924_150351_1_final_bao_vd_-0.006_lmp_-0.01_ltp_-0.01_fdr_10_wb_3_run_0/outputs"
#seed 9
default_dir = "/srv/marl/nathanwu/rmappo-MultiAgentForagingEnv-check/20250924_150413_1_final_bao_vd_-0.006_lmp_-0.01_ltp_-0.01_fdr_10_wb_3_run_9/outputs"
#seed 8
default_dir = "/srv/marl/nathanwu/rmappo-MultiAgentForagingEnv-check/20250924_150409_1_final_bao_vd_-0.006_lmp_-0.01_ltp_-0.01_fdr_10_wb_3_run_8/outputs"
#0wb best seed
default_dir = "/srv/marl/nathanwu/rmappo-MultiAgentForagingEnv-check/20250923_202836_1_bao_vd_-0.006_lmp_-0.01_ltp_-0.01_fdr_10_wb_0_run_8/outputs"

outputs_folder = "./results/rmappo-MultiAgentForagingEnv-check/20250808_153214_1_bao_efp_0.05_vd_0.002_fd_10/outputs"
outputs_folder = ru.get_latest_outputs_folder(default_dir)

# outputs_folder = "/home/raaghav/zfish/onpolicy/custom/forage/results/rmappo-MultiAgentForagingEnv-check/20250808_153214_1_bao_efp_0.05_vd_0.002_fd_10/additional_exps"

# ru.get_latest_outputs_folder(default_dir)
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


# In[ ]:


# Create a figure results folder
outputs_folder = f"{outputs_folder}/figures"
os.makedirs(outputs_folder, exist_ok=True)

print(f"Created folder: {outputs_folder}")

if not os.access(outputs_folder, os.W_OK):
    print("Cannot write to folder: results folder is /srv/marl/nathanwu/outputs/figures instead")

outputs_folder = "/home/nathanwu/outputs/test_outputs/figures"


# In[ ]:


dff.drop(columns=["rnn_states"], inplace=True)


# In[ ]:


dff = dff.sort_values(
    by=["env_id", "episode_index", "agent_id", "time_step"]
).reset_index(drop=True)

print(dff.head())


# In[ ]:


dff["actions"]


# In[ ]:


# dff["move_forward"] = dff["move_forward"] * cfg.FISH_CONSTANTS["max_speed"]
# dff["turn_angle"] = dff["turn_angle"] * cfg.FISH_CONSTANTS["max_turn_speed"]

# # Compute vergence angle and speed
# perception_field = 163 * np.pi / 180
# dff['vergence_angle'] = dff['left_eye_angle'] - dff['right_eye_angle'] + perception_field
# dff['speed'] = dff['displacement'] * cfg.ENV_PARAMS["fps_sim"]


# In[ ]:


# # Speed is calculated via displacement so has some bugs -- more interesting to look at move_forward

# dff['speed'] = dff['move_forward']
# dff['vergence_angle_deg'] = dff['vergence_angle'] * 180 / np.pi


# In[ ]:


from utils_behavior import analyze_vergence_during_food_tracking

# Perform the analysis
tracking_sequences_df = analyze_vergence_during_food_tracking(dff)


# In[ ]:



from utils_behavior import plot_first_success_story
import os

# usage:
success_num = 17
first_success = tracking_sequences_df[tracking_sequences_df["outcome"] == "success"].iloc[success_num]
save_dir = os.path.join(outputs_folder, "report_figures")
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, "first_success_story.png")

fig, ax = plot_first_success_story(dff, first_success, save_path=save_path)


# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patheffects as pe
import seaborn as sns
from utils_figstyle import *

def plot_first_success_kinematics(dff, tracking_results, first_success, miss=False, pre_steps=20, post_steps=20, save_path=None, dpi=300, fz=8):
    """
    Time-series story for the first successful hunt:
      - move_forward (top) and orientation (bottom)
      - split into pre-hunt [t in (-pre_steps..0)], hunting [0..capture], post-hunt [capture..+post_steps]
      - inline labels, no legend; vertical markers at t=0 and at capture
    """
    #set_nature_style()
    # --- pick the first success ---
    env_id         = first_success["env_id"]
    episode_index  = first_success["episode_index"]
    agent_id       = first_success["agent_id"]
    start_time     = int(first_success["start_time_step"])
    duration       = int(first_success["tracking_duration"])
    hunt_end       = start_time + duration

    # episode slice for this agent (sorted)
    traj = dff[(dff["env_id"]==env_id) &
               (dff["episode_index"]==episode_index) &
               (dff["agent_id"]==agent_id)].sort_values("time_step")

    # time windows
    tmin_abs = max(0, start_time - pre_steps)
    # allow post window up to either end of episode or desired steps
    episode_tmax = int(traj["time_step"].max())
    post_end_abs = min(hunt_end + post_steps, episode_tmax)

    window = traj[(traj["time_step"] >= tmin_abs) & (traj["time_step"] <= post_end_abs)].copy()
    if window.empty:
        print("Warning: empty window for first success; nothing to plot.")
        return

    # relative time axis (t=0 at detection/lock-on)
    window["t_rel"] = window["time_step"] - start_time

    # extract series
    if not {"move_forward","orientation"}.issubset(window.columns):
        raise KeyError("dff must contain 'move_forward' and 'orientation' columns.")

    t   = window["t_rel"].to_numpy()
    mv  = window["move_forward"].to_numpy()
    trn = window["turn_angle"].to_numpy()

    trn = trn * 180 / np.pi  # convert to degrees
 
    # # Get the target food ID from first_success
    # target_food_id = first_success["food_id"]

    # orient_to_food_deg = []
    # for _, row in window.iterrows():
    #     agent_pos_t = np.array(row["position"])
    #     agent_orientation = row["orientation"]

    #     # Find target food position in this timestep (if present)
    #     if target_food_id in row["food_ids"]:
    #         idx = row["food_ids"].index(target_food_id)
    #         target_food_pos = np.array(row["food_positions"][idx])

    #         # Vector from agent → target food
    #         food_vector = target_food_pos - agent_pos_t
    #         food_direction = np.arctan2(food_vector[1], food_vector[0])

    #         # Angular difference wrapped to [-pi, pi]
    #         angular_diff = food_direction - agent_orientation
    #         angular_diff = np.arctan2(np.sin(angular_diff), np.cos(angular_diff))

    #         orient_to_food_deg.append(angular_diff * 180 / np.pi)
    #     else:
    #         # If target food not present, append NaN
    #         orient_to_food_deg.append(np.nan)

    # orient_to_food_deg = np.array(orient_to_food_deg)

    # masks
    pre_mask    = t <= 0
    hunt_mask   = (t >= 0) & (window["time_step"] <= hunt_end)
    post_mask   = window["time_step"] >= hunt_end

    # Okabe–Ito
    c_explore = "#0072B2"   # blue
    c_hunt    = "#E69F00"   # orange
    c_post    = "#0072B2"   # blue
    vline_col = "#4D4D4D"

    def _plot_segments(ax, x, y):
        # plot each segment with its color
        if pre_mask.any():
            ax.plot(x[pre_mask],  y[pre_mask],  color=c_explore, lw=2.0,
                    path_effects=[pe.Stroke(linewidth=2.6, foreground="white"), pe.Normal()])
        if hunt_mask.any():
            ax.plot(x[hunt_mask], y[hunt_mask], color=c_hunt,    lw=2.2,
                    path_effects=[pe.Stroke(linewidth=3.0, foreground="white"), pe.Normal()])
        if post_mask.any():
            ax.plot(x[post_mask], y[post_mask], color=c_post,    lw=2.0,
                    path_effects=[pe.Stroke(linewidth=2.6, foreground="white"), pe.Normal()])

    def _inline_label_xaxis(ax, x, mask, text, color, dy_frac=-0.08, fz=8):
        """
        Place a label along the x-axis under the segment's midpoint.
        dy_frac is the offset as a fraction of the y-axis range.
        """
        idxs = np.flatnonzero(mask)
        if idxs.size == 0:
            return
        mid_x = x[idxs[len(idxs)//2]]
        ymin, ymax = ax.get_ylim()
        y_range = ymax - ymin
        y_pos = ymin + float(dy_frac) * float(y_range)  # place slightly below the bottom
        ax.text(mid_x, y_pos, text, fontsize=fz, color=color,
                ha='center', va='top', clip_on=False)


    fig, axes = plt.subplots(2, 1, figsize=(6, 4.6), sharex=True)

    # top: move_forward
    ax0 = axes[0]
    _plot_segments(ax0, t, mv)
    _inline_label_xaxis(ax0, t, pre_mask,  "pre‑hunt",   c_explore, fz=fz)
    _inline_label_xaxis(ax0, t, hunt_mask, "hunting",    c_hunt, fz=fz)
    _inline_label_xaxis(ax0, t, post_mask, "post‑hunt",  c_post, fz=fz)
    ax0.set_ylabel("Forward speed (mm/s)", fontsize=fz)
    ax0.grid(True, linewidth=0.6, alpha=0.2)

    # bottom: turn_angle
    ax1 = axes[1]
    _plot_segments(ax1, t, trn)
    # _inline_label_xaxis(ax1, t, pre_mask,  "pre‑hunt",   c_explore)
    # _inline_label_xaxis(ax1, t, hunt_mask, "hunting",    c_hunt)
    # _inline_label_xaxis(ax1, t, post_mask, "post‑hunt",  c_post)
    ax1.set_ylabel("Turn speed (°/s)", fontsize=fz)
    ax1.set_xlabel("Time (seconds)", fontsize=fz)
    ax1.grid(True, linewidth=0.6, alpha=0.2)

    # scale x-axis by 0.2
    # Label each second in whole numbers
    scale = 1 / cfg.ENV_PARAMS["fps_sim"]
    ticks = np.arange(np.ceil(t.min() * scale), np.floor(t.max() * scale) + 1, 1)
    axes[-1].set_xticks(ticks / scale)
    axes[-1].set_xticklabels([f"{int(x)}" for x in ticks])

    # vertical markers: t=0 (detect) and capture
    for ax in axes:
        ax.tick_params(axis='both', which='major', labelsize=fz)
        ax.axvline(0,        color=vline_col, lw=1.0, ls="--")
        ax.axvline(duration, color=vline_col, lw=1.0, ls=":",  alpha=0.8)
        # small labels (no boxes/arrows)
        ax.text(0 + 0.2,        ax.get_ylim()[1]*1.05, "detect",  color=vline_col, fontsize=fz, ha='left', va='top')
        if not miss:
            ax.text(duration + 0.2, ax.get_ylim()[1]*1.05, "strike", color=vline_col, fontsize=fz, ha='left', va='top')
        else:
            ax.text(duration + 0.2, ax.get_ylim()[1]*1.05, "miss", color=vline_col, fontsize=fz, ha='left', va='top')

        # tidy spines
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # x limits to window
    axes[-1].set_xlim(t.min(), t.max())

    sns.despine(ax=ax)
    fig.align_ylabels(axes)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=dpi)
    return fig, axes

save_path = os.path.join(save_dir, "first_success_kinematics.png")
# usage:
fig, axes = plot_first_success_kinematics(dff, tracking_sequences_df, first_success, save_path=save_path, fz=14)


# In[ ]:


success_num = 44
first_miss = tracking_sequences_df[tracking_sequences_df["outcome"] == "miss"].iloc[success_num]
save_path = os.path.join(save_dir, "first_miss_story.png")
fig, ax = plot_first_success_story(dff, first_miss, is_miss=True, save_path=save_path, prey_pre_dx_dy=(-40, 10))


# In[ ]:


save_path = os.path.join(save_dir, "first_miss_kinematics.png")
fig, axes = plot_first_success_kinematics(dff, tracking_sequences_df, first_miss, miss=True, save_path=save_path, fz=14)


# In[ ]:


all_trajectories = tracking_sequences_df[(tracking_sequences_df["outcome"] == "success") | (tracking_sequences_df["outcome"] == "miss")]


# In[ ]:


# Filter success trajectories
fz = 15
# Extract move_forward values for success trajectories
#set_nature_style()
success_time_steps = []
success_turn_angles = []
for _, row in all_trajectories.iterrows():
    env_id = row["env_id"]
    episode_index = row["episode_index"]
    agent_id = row["agent_id"]
    start_time = row["start_time_step"]
    duration = row["tracking_duration"]
    success_time_steps.extend(
        dff[
            (dff["env_id"] == env_id) &
            (dff["episode_index"] == episode_index) &
            (dff["agent_id"] == agent_id) &
            (dff["time_step"] >= start_time) &
            (dff["time_step"] < start_time + duration)
        ]["move_forward"].values
    )
    success_turn_angles.extend(
        dff[
            (dff["env_id"] == env_id) &
            (dff["episode_index"] == episode_index) &
            (dff["agent_id"] == agent_id) &
            (dff["time_step"] >= start_time) &
            (dff["time_step"] < start_time + duration)
        ]["turn_angle"].values
    )

# Extract move_forward and turn_angle values for time steps not in miss or success
all_tracked_time_steps = []
for _, row in tracking_sequences_df.iterrows():
    env_id = row["env_id"]
    episode_index = row["episode_index"]
    agent_id = row["agent_id"]
    start_time = row["start_time_step"]
    duration = row["tracking_duration"]
    all_tracked_time_steps.extend(
        dff[
            (dff["env_id"] == env_id) &
            (dff["episode_index"] == episode_index) &
            (dff["agent_id"] == agent_id) &
            (dff["time_step"] >= start_time) &
            (dff["time_step"] < start_time + duration)
        ].index
    )

non_tracked_time_steps = dff.loc[~dff.index.isin(all_tracked_time_steps), "move_forward"].values
non_tracked_turn_angles = dff.loc[~dff.index.isin(all_tracked_time_steps), "turn_angle"].values

# Plot histograms
fig, axes = plt.subplots(1, 2, figsize=(5.5, 1.8), sharey=True)
# fig, axes = plt.subplots(2, 1, figsize=(7, 10), sharex=False)


palette = sns.color_palette("Set1", 2)
c_hunting, c_nonhunting = palette[0], palette[1]
# Plot for move_forward
# axes[0].hist(success_time_steps, bins=30, alpha=0.7, label="Hunting", color=c_hunting, density=True)
# axes[0].hist(non_tracked_time_steps, bins=30, alpha=0.7, label="Non-Hunting", color=c_nonhunting, density=True)
sns.kdeplot(success_time_steps, color=c_hunting, lw=2, ax=axes[0])
sns.kdeplot(non_tracked_time_steps, color=c_nonhunting, lw=2, ax=axes[0])
axes[0].set_xlabel("Forward speed (mm/s)", fontsize=fz)
axes[0].set_ylabel("Frequency", fontsize=fz)
axes[0].set_xlim(0, cfg.FISH_CONSTANTS["max_speed"])
axes[0].axvline(cfg.AGENT_PARAMS["penalize_move_threshold"] * cfg.FISH_CONSTANTS["max_speed"], color="black", linestyle="--", linewidth=1.5, label=r"$v_{\mathrm{threshold}}$")
#axes[0].legend(fontsize=fz, loc='upper right')
#axes[0].grid(alpha=0.3)
# axes[0].set_title("Move Forward Distribution", fontsize=20)

# Plot for turn_angle
# axes[1].hist(success_turn_angles, bins=30, alpha=0.7, label="Hunting", color=c_hunting, density=True)
# axes[1].hist(non_tracked_turn_angles, bins=30, alpha=0.7, label="Non-Hunting", color=c_nonhunting, density=True)
sns.kdeplot(np.abs(success_turn_angles), color=c_hunting, lw=2, ax=axes[1], label="Hunting")
sns.kdeplot(np.abs(non_tracked_turn_angles), color=c_nonhunting, lw=2, ax=axes[1], label="Non-Hunting")
axes[1].set_xlabel("Turn speed (rad/s)", fontsize=fz)
axes[1].axvline(cfg.AGENT_PARAMS["penalize_turn_threshold"] * cfg.FISH_CONSTANTS["max_turn_speed"], color="black", linestyle="--", linewidth=1.5, label=r"Penalty Threshold")
# axes[1].set_ylabel("Frequency", fontsize=fz)
# axes[1].set_xlim(-cfg.FISH_CONSTANTS["max_turn_speed"], cfg.FISH_CONSTANTS["max_turn_speed"])
axes[1].set_xlim(0, cfg.FISH_CONSTANTS["max_turn_speed"])
#axes[1].grid(alpha=0.3)
# axes[1].set_title("Turn Angle Distribution", fontsize=20)
for ax in axes:
    ax.tick_params(axis='both', which='major', labelsize=fz)

sns.despine(ax=axes[0])
sns.despine(ax=axes[1])

plt.tight_layout()
axes[1].legend(loc='lower right', fontsize=fz, ncol = 3, bbox_to_anchor = [1.2, 1], columnspacing=1, handlelength=1)

save_path = os.path.join(save_dir, "success_vs_non_hunting_histograms.png")
plt.savefig(save_path, bbox_inches="tight", dpi=300)
plt.show()


# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import transforms
import seaborn as sns
from utils_behavior import analyze_distance_angle_to_consumed_food

fz = 15

# ---------- small helpers ----------
def _apply_pub_style(dpi=300):
    plt.rcParams.update({
        "font.size": 10, "axes.labelsize": 11,
        "xtick.labelsize": 10, "ytick.labelsize": 10,
        "savefig.dpi": dpi, "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.spines.top": False, "axes.spines.right": False
    })

def _top_phase_label(ax, x_min, x_max, text, color, y_axes=1.05, fz=9, weight='bold'):
    """Phase label centered between x_min and x_max, just above the top subplot."""
    if x_max < x_min:  # handle degenerate single-point case
        x_max = x_min
    mid_x = (x_min + x_max) / 2.0
    trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
    ax.text(mid_x, y_axes, text, transform=trans, ha="center", va="bottom",
            color=color, fontsize=fz, fontweight=weight, clip_on=False)

# ---------- main ----------
def plot_distance_angle_stacked(distance_angle_results, cfg, fps=None, dpi=300):
    """
    Two vertically-stacked panels that share x-axis:
      Top: Distance to consumed prey (mean ± SEM) + sensing/eating radii.
      Bottom: Angle to consumed prey (mean ± SEM) + 0° and ±½ eating-angle.
    t=0 'Eating' marker drawn on BOTH axes so the line looks continuous.
    Phase labels ('Pre-eating approach', 'Eating', 'Post-eating') shown at the TOP.
    """
    _apply_pub_style(dpi)

    # --- summarize mean ± SEM by relative time ---
    summary = (distance_angle_results
               .groupby('time_relative')
               .agg(mean_distance=('distance_to_consumed_food', 'mean'),
                    std_distance =('distance_to_consumed_food', 'std'),
                    n            =('distance_to_consumed_food', 'count'),
                    mean_angle   =('angle_to_consumed_food_deg', 'mean'),
                    std_angle    =('angle_to_consumed_food_deg', 'std'))
               .reset_index())
    summary['sem_distance'] = summary['std_distance'] / np.sqrt(summary['n']).replace(0, np.nan)
    summary['sem_angle']    = summary['std_angle']    / np.sqrt(summary['n']).replace(0, np.nan)

    # x in steps or seconds
    t_steps = summary['time_relative'].to_numpy()
    if fps and fps > 0:
        x = t_steps / float(fps)
        x_label = "Time (s)"
    else:
        x = t_steps
        x_label = "Time steps relative to eating (t = 0)"

    # Okabe–Ito palette
    C_DIST  = "#0072B2"   # blue (distance)
    C_ANG   = "#009E73"   # green (angle)
    C_EVENT = "#D55200"   # eating (t=0)
    C_GUIDE = "#4D4D4D"   # gray guides/text
    C_BAND  = "#56A2E9"   # phase labels

    m_dist, se_dist = summary['mean_distance'].to_numpy(), summary['sem_distance'].to_numpy()
    m_ang,  se_ang  = np.abs(summary['mean_angle'].to_numpy()),    summary['sem_angle'].to_numpy()

    # --- figure: two rows, shared x ---
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(7, 4), sharex=True,
        gridspec_kw=dict(hspace=0.08)
    )

    # ========== TOP: Distance ==========
    ax_top.plot(x, m_dist, color=C_DIST, lw=2.2)
    ax_top.fill_between(x, m_dist - se_dist, m_dist + se_dist, color=C_DIST, alpha=0.18)
    ax_top.set_ylabel(r"$\mathrm{d_{prey}}$ (mm)", labelpad=20, fontsize=fz)

    # Sensing / eating radii
    fr = float(cfg.AGENT_PARAMS["food_detection_range"])
    er = float(cfg.AGENT_PARAMS["eating_radius"])
    ax_top.axhline(fr, color=C_GUIDE, lw=1.0, ls=":", alpha=0.7)
    ax_top.axhline(er, color=C_GUIDE, lw=1.0, ls=":", alpha=0.7)
    # small inline labels
    x_left = float(np.nanmin(x) + 0.05*(np.nanmax(x)-np.nanmin(x)))
    ax_top.text(x_left, er, "strike radius", color=C_GUIDE, fontsize=fz, va='bottom', ha='left')
    ax_top.text(x_left, fr, "sensing radius", color=C_GUIDE, fontsize=fz, va='top', ha='left')
    # ax_top.text(x_left, er, "strike radius",  color=C_GUIDE, fontsize=fz, va='bottom', ha='left')

    # ========== BOTTOM: Angle ==========
    ax_bot.plot(x, m_ang, color=C_ANG, lw=2.2)
    ax_bot.fill_between(x, m_ang - se_ang, m_ang + se_ang, color=C_ANG, alpha=0.18)
    ax_bot.set_ylabel(r"$\mathrm{\theta_{prey}}$ (°)", labelpad=20, fontsize=fz)
    ax_bot.set_xlabel(x_label, fontsize=fz)

    ea_half_deg = float(cfg.AGENT_PARAMS["eating_angle"]) * 180.0/np.pi / 2.0
    ax_bot.axhline(0,            color=C_GUIDE, lw=1.0, ls=":", alpha=0.7)
    # ax_bot.axhline(+ea_half_deg, color=C_GUIDE, lw=1.0, ls=":", alpha=0.7)
    # ax_bot.axhline(-ea_half_deg, color=C_GUIDE, lw=1.0, ls=":", alpha=0.7)
    ax_bot.text(x_left, 0,            "direct alignment", color=C_GUIDE, fontsize=fz, va='bottom', ha='left')
    # ax_bot.text(x_left, +ea_half_deg, "eating angle +½",  color=C_GUIDE, fontsize=20, va='bottom', ha='left')

    # ========== Shared t=0 "Eating" marker across BOTH axes ==========
    for ax in (ax_top, ax_bot):
        ax.axvline(0, color=C_EVENT, lw=1.2, ls='--')  # draw on both → visually continuous

    # ========== Phase labels at the TOP (on the top axis) ==========
    xmin, xmax = float(np.nanmin(x)), float(np.nanmax(x))
    if np.any(x < 0):
        _top_phase_label(ax_top, xmin, 0.0, "Pre‑strike approach", C_BAND, fz=fz, y_axes=1.05)
    _top_phase_label(ax_top, 0.0, 0.0, "Strike", C_EVENT, fz=fz, y_axes=1.05)
    if np.any(x > 0):
        _top_phase_label(ax_top, 0.0, xmax, "Post‑strike", C_BAND, y_axes=1.05)

    # # Quiet N info
    # n_events = distance_angle_results["eating_time_step"].nunique()
    # ax_top.text(0.01, 0.98, f"N = {n_events} eating events",
    #             transform=ax_top.transAxes, ha="left", va="top",
    #             fontsize=9, color=C_GUIDE)

    # Set x and y ticks fontsize
    ax_top.tick_params(axis='both', which='major', labelsize=fz)
    ax_bot.tick_params(axis='both', which='major', labelsize=fz)

    # Tidy
    for ax in (ax_top, ax_bot):
        ax.grid(True, alpha=0.25, linewidth=0.6)
    sns.despine(ax=ax_top); sns.despine(ax=ax_bot)

    fig.align_ylabels([ax_top, ax_bot])


    fig.tight_layout()
    return fig, (ax_top, ax_bot)

# ----- usage -----
distance_angle_results = analyze_distance_angle_to_consumed_food(dff, window_size=80)
fig, (ax_top, ax_bot) = plot_distance_angle_stacked(distance_angle_results, cfg, fps=cfg.ENV_PARAMS["fps_sim"], dpi=300)
save_path = os.path.join(save_dir, "distance_angle_stacked.png")
fig.savefig(save_path, bbox_inches="tight")


# In[ ]:


distance_angle_results


# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

fontsize=15

# Use seaborn style
#sns.set(style="whitegrid", font_scale=1.2)

# Prepare data (remove durations == 1)
success_data = tracking_sequences_df[(tracking_sequences_df['outcome'] == 'success') &
                                (tracking_sequences_df['tracking_duration'] != 1)]
miss_data = tracking_sequences_df[(tracking_sequences_df['outcome'] == 'miss') &
                             (tracking_sequences_df['tracking_duration'] != 1)]

success_durations = success_data['tracking_duration'].values / cfg.ENV_PARAMS["fps_sim"]
miss_durations = miss_data['tracking_duration'].values / cfg.ENV_PARAMS["fps_sim"]

# Log-spaced bins
# min_duration = min(success_durations.min(), miss_durations.min())
# max_duration = max(success_durations.max(), miss_durations.max())
# log_bins = np.logspace(np.log10(max(1, min_duration)), np.log10(max_duration), 20)

# Pick colors from seaborn palette
palette = sns.color_palette("Set2", 2)
c_miss, c_success = palette[1], palette[0]

# Create figure
fig, ax = plt.subplots(figsize=(2.5, 1.8))

# Plot normalized histograms (densities)
# Plot normalized histograms (densities) ensuring higher bars are under lower bars
# ax.hist(
#     miss_durations, bins=log_bins,
#     alpha=0.6, density=True,
#     label=f"Miss",
#     color=c_miss, edgecolor='black', linewidth=0.5,
#     zorder=1  # Lower z-order to ensure it's under the success bars
# )
# ax.hist(
#     success_durations, bins=log_bins,
#     alpha=0.4, density=True,
#     label=f"Success", edgecolor='black',
#     color=c_success, linewidth=0.5,
#     zorder=2  # Higher z-order to ensure it's above the miss bars
# )

clip = [1, np.inf]
sns.kdeplot(miss_durations, color=c_miss, lw=2, ax=ax, clip = clip, label = "Miss")
sns.kdeplot(success_durations, color=c_success, lw=2, ax=ax, clip = clip, label = "Success")

# Log scale on x-axis
ax.set_xscale('log')

# Labels and title
ax.set_xlabel('Tracking duration (s)', fontsize=fontsize)
ax.set_ylabel('Frequency', fontsize=fontsize)

# Legend and grid
ax.tick_params(axis='both', which='major', labelsize=fontsize)
# ax.grid(True, which="both", axis="y", alpha=0.3)
sns.despine(ax=ax)

plt.tight_layout()
ax.legend(loc='lower center', fontsize=fontsize, ncol = 2, bbox_to_anchor = [0.5, 1], columnspacing=1, handlelength=1)
plt.savefig(os.path.join(save_dir, "success_vs_miss_durations_histogram.png"), bbox_inches="tight", dpi=300)
plt.show()


# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

fontsize=30

# Use seaborn style
#sns.set(style="whitegrid", font_scale=1.2)

# Prepare data (remove durations == 1)
success_data = tracking_sequences_df[(tracking_sequences_df['outcome'] == 'success') &
                                (tracking_sequences_df['tracking_duration'] != 1)]
miss_data = tracking_sequences_df[(tracking_sequences_df['outcome'] == 'miss') &
                             (tracking_sequences_df['tracking_duration'] != 1)]

success_durations = success_data['tracking_duration'].values / cfg.ENV_PARAMS["fps_sim"]
miss_durations = miss_data['tracking_duration'].values / cfg.ENV_PARAMS["fps_sim"]

# Log-spaced bins
min_duration = min(success_durations.min(), miss_durations.min())
max_duration = max(success_durations.max(), miss_durations.max())
log_bins = np.logspace(np.log10(max(1, min_duration)), np.log10(max_duration), 20)

# Colors
palette = sns.color_palette("Set2", 2)
c_miss, c_success = palette[1], palette[0]

# Compute densities with shared bins
miss_counts, edges = np.histogram(miss_durations, bins=log_bins, density=True)
succ_counts, _     = np.histogram(success_durations, bins=log_bins, density=True)
widths = np.diff(edges)

# Figure
fig, ax = plt.subplots(figsize=(5, 5))

# Draw each bin: taller first (under), shorter second (on top)
for i, (hm, hs) in enumerate(zip(miss_counts, succ_counts)):
    left = edges[i]
    w    = widths[i]

    # define two bars
    miss_bar = dict(x=left, height=hm, width=w, color=c_miss, alpha=0.6, edgecolor='black', linewidth=0.5)
    succ_bar = dict(x=left, height=hs, width=w, color=c_success, alpha=0.6, edgecolor='black', linewidth=0.5)

    if hm >= hs:
        ax.bar(**miss_bar, align='edge', zorder=1)
        ax.bar(**succ_bar, align='edge', zorder=2)
    else:
        ax.bar(**succ_bar, align='edge', zorder=1)
        ax.bar(**miss_bar, align='edge', zorder=2)

# Log scale on x-axis
ax.set_xscale('log')

# Labels and title
ax.set_xlabel('Tracking duration (s)', fontsize=fontsize)
ax.set_ylabel('Density', fontsize=fontsize)

# Legend (add proxy handles so labels match colors/alphas)
from matplotlib.patches import Patch
handles = [
    Patch(facecolor=c_miss, edgecolor='black', linewidth=0.5, alpha=0.6, label='Miss'),
    Patch(facecolor=c_success, edgecolor='black', linewidth=0.5, alpha=0.6, label='Success'),
]
ax.legend(handles=handles, frameon=False, loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=2, fontsize=fontsize)

# Ticks/grid
ax.tick_params(axis='both', which='major', labelsize=fontsize)
ax.grid(True, which="both", axis="y", alpha=0.3)
sns.despine(ax=ax)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, "success_vs_miss_durations_histogram_bars.png"), bbox_inches="tight", dpi=300)
plt.show()


# In[ ]:





# In[ ]:




