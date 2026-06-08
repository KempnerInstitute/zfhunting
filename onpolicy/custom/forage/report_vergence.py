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

default_dir = "./results/rmappo-MultiAgentForagingEnv-check/"
if not os.path.exists(default_dir): # Running on cluster
    default_dir = "/n/home04/ramalik/ZFish/onpolicy/custom/forage/results/rmappo-MultiAgentForagingEnv-check"

# outputs_folder = "./results/rmappo-MultiAgentForagingEnv-check/20250808_153214_1_bao_efp_0.05_vd_0.002_fd_10/outputs"
outputs_folder = "results/rmappo-MultiAgentForagingEnv-check/20250909_140455_1_bao_vd_0.003_fd_10_action_noise_0.1/outputs/"
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


dff.drop(columns=["rnn_states"], inplace=True)


# In[ ]:


dff = dff.sort_values(
    by=["env_id", "episode_index", "agent_id", "time_step"]
).reset_index(drop=True)

print(dff.head())


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


from utils_behavior import calculate_avg_vergence_by_outcome

success_trajectories, miss_trajectories = calculate_avg_vergence_by_outcome(tracking_sequences_df)


# In[ ]:


import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils_behavior import non_tracking_data, calculate_auc_hunting_no_hunting
from utils_figstyle import *

# ----- data prep -----
set_nature_style()
sns.set(style="whitegrid", font_scale=1.2)  # compatible with seaborn 0.10.x

non_tracking_vergence = non_tracking_data(dff, tracking_sequences_df)
avg_auc, non_tracking_auc = calculate_auc_hunting_no_hunting(success_trajectories, non_tracking_vergence)

# flatten success vergence angles
tracking_vergence = np.array([ang for _, angles in success_trajectories for ang in angles])

# use absolute vergence angles (to match your original)
trk = np.abs(tracking_vergence)
non = np.abs(np.asarray(non_tracking_vergence))

# ----- figure -----
fig, ax = plt.subplots(figsize=(7.5, 5.2))

# seaborn palette
palette = sns.color_palette("Set1", 3)
c_hunt = palette[2]
c_non = palette[1]
# sns.color_palette("colorblind", 2)

# KDEs (shade=True for seaborn<=0.10)
sns.kdeplot(trk, ax=ax, shade=True, alpha=0.25, lw=2, color=c_hunt, label=f"Food tracking (success)")
sns.kdeplot(non, ax=ax, shade=True, alpha=0.25, lw=2, color=c_non, label=f"Non-tracking")

trk_std = np.std(trk)
non_std = np.std(non)

max_vergence = (cfg.FISH_CONSTANTS["max_left_vergence"] - cfg.FISH_CONSTANTS["max_right_vergence"] + cfg.FISH_CONSTANTS["perception_field"]) * 180 / np.pi
min_vergence = (cfg.FISH_CONSTANTS["min_left_vergence"] - cfg.FISH_CONSTANTS["min_right_vergence"] + cfg.FISH_CONSTANTS["perception_field"]) * 180 / np.pi

avg_auc = avg_auc * max_vergence
non_tracking_auc = non_tracking_auc * max_vergence

# subtle AUC callouts
# subtle AUC callouts repositioned under the legend
ax.text(0.5, 0.8, f"Avg vergence: {avg_auc:.2f}° ± {trk_std:.2f}°", transform=ax.transAxes,
        ha="left", va="top", fontsize=15, color=c_hunt, alpha=1)
ax.text(0.5, 0.72, f"Avg vergence: {non_tracking_auc:.2f}° ± {non_std:.2f}°", transform=ax.transAxes,
        ha="left", va="top", fontsize=15, color=c_non, alpha=1)

# labels & style
ax.set_xlabel("Vergence angle (°)")
ax.set_ylabel("Density")
# no heavy title—keep it clean for a paper
ax.legend(frameon=False, loc="upper right", fontsize=15)
sns.despine(ax=ax)
ax.grid(alpha=0.25, linewidth=0.6)

plt.tight_layout()
save_dir = os.path.join(outputs_folder, "report_figures")
os.makedirs(save_dir, exist_ok=True)
plt.savefig(os.path.join(save_dir, "success_vs_non_hunting_vergence_histogram.png"), bbox_inches="tight", dpi=300)
plt.show()


# In[ ]:


# success_data = tracking_results[tracking_results['outcome'] == 'success']
# miss_data = tracking_results[tracking_results['outcome'] == 'miss']
success_data, miss_data = calculate_avg_vergence_by_outcome(tracking_sequences_df)

fontsize = 20

# Plot sample trajectories for successful hunts
fig, ax = plt.subplots(figsize=(6, 4))

for i, (times, angles) in enumerate(success_data[:5]):
    ax.plot(times, angles, alpha=0.7, linewidth=1, label=f"Trajectory {i+1}")

ax.set_xlabel('Normalized Time', fontsize=fontsize)
ax.set_ylabel('Vergence Angle (°)', fontsize=fontsize)
ax.set_title('Successful Hunts', fontsize=fontsize)
ax.grid(True, alpha=0.3)
ax.tick_params(axis='both', which='major', labelsize=fontsize)

sns.despine(ax=ax)

plt.tight_layout()
plt.show()

# Plot sample trajectories for failed hunts
fig, ax = plt.subplots(figsize=(6, 4))

for i, (times, angles) in enumerate(miss_data[:5]):
    ax.plot(times, angles, alpha=0.7, linewidth=1, label=f"Trajectory {i+1}")

ax.set_xlabel('Normalized Time', fontsize=fontsize)
ax.set_ylabel('Vergence Angle (°)', fontsize=fontsize)
ax.set_title('Failed Hunts', fontsize=fontsize)
ax.grid(True, alpha=0.3)
ax.tick_params(axis='both', which='major', labelsize=fontsize)

sns.despine(ax=ax)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, "sample_miss_trajectories.png"), bbox_inches="tight", dpi=300)
plt.show()


# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

fz = 25

# Style (works with older seaborn)
sns.set(style="whitegrid", font_scale=1.2)

# Filter for success outcomes
success_tracking_results = tracking_sequences_df[tracking_sequences_df['outcome'] == 'success']

# Data
success_tracking_results['monocular_frequency'] = (
    success_tracking_results['detected_frequency'] - success_tracking_results['binocular_frequency']
)
binoc = success_tracking_results['binocular_frequency'].to_numpy()
monoc = success_tracking_results['monocular_frequency'].to_numpy()

# Shared bins across both distributions
xmin = np.nanmin([binoc.min(), monoc.min()])
xmax = np.nanmax([binoc.max(), monoc.max()])
bins = np.linspace(xmin, xmax, 50)

# Colors (colorblind-friendly)
c_binoc, c_monoc = sns.color_palette("colorblind", 2)

fig, ax = plt.subplots(figsize=(7.5, 5.2))

# Plot: binocular filled, monocular outline → no blended “third color”
ax.hist(
    binoc, bins=bins, color=c_binoc, alpha=0.6, edgecolor="black", linewidth=0.5,
    label=f"Binocular", density=True
)
ax.hist(
    monoc, bins=bins, color=c_monoc, linewidth=0.5,
    label=f"Monocular", alpha=0.6, density=True, edgecolor="black"
)

# Labels & styling
ax.set_xlabel("Frequency", fontsize=fz)
ax.set_ylabel("Density", fontsize=fz)
ax.legend(frameon=False, loc="upper right", fontsize=fz)
ax.tick_params(axis='both', which='major', labelsize=fz)
ax.grid(True, axis="y", alpha=0.3, linewidth=0.6)
sns.despine(ax=ax)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, "success_vs_non_hunting_histograms.png"), bbox_inches="tight", dpi=300)
plt.show()


# In[ ]:


from utils_behavior import analyze_vergence_speed_three_phases

set_nature_style()
fontsize = 25
# Perform the analysis
phase_duration = 10  # Duration for pre and post phases
phase_data = analyze_vergence_speed_three_phases(tracking_sequences_df, dff, phase_duration=phase_duration)

# Softer red/green colors (muted style)
palette = sns.color_palette("muted")  
c_non = palette[3]    # soft red
c_hunt = palette[2]   # soft green


# Separate by outcome
success_phases = [d for d in phase_data if d['outcome'] == 'success']
miss_phases = [d for d in phase_data if d['outcome'] == 'miss']

print(f"Success trajectories: {len(success_phases)}")
print(f"Miss trajectories: {len(miss_phases)}")

# Create common time grid for interpolation
time_grid = np.linspace(-1, 2, 150)  # 150 points from -1 to 2

# Interpolate trajectories
def interpolate_phases(trajectories, time_grid):
    interpolated_vergence = []
    interpolated_speed = []
    
    for traj in trajectories:
        # Interpolate vergence
        vergence_interp = np.interp(time_grid, traj['time'], traj['vergence'])
        interpolated_vergence.append(vergence_interp)
        
        # Interpolate speed
        speed_interp = np.interp(time_grid, traj['time'], traj['speed'])
        interpolated_speed.append(speed_interp)
    
    return np.array(interpolated_vergence), np.array(interpolated_speed)

# Interpolate for both outcomes
success_vergence, success_speed = interpolate_phases(success_phases, time_grid)
miss_vergence, miss_speed = interpolate_phases(miss_phases, time_grid)

# Calculate means and standard errors
success_vergence_mean = np.mean(success_vergence, axis=0)
success_vergence_sem = np.std(success_vergence, axis=0) / np.sqrt(len(success_phases))
success_speed_mean = np.mean(success_speed, axis=0)
success_speed_sem = np.std(success_speed, axis=0) / np.sqrt(len(success_phases))

miss_vergence_mean = np.mean(miss_vergence, axis=0)
miss_vergence_sem = np.std(miss_vergence, axis=0) / np.sqrt(len(miss_phases))
miss_speed_mean = np.mean(miss_speed, axis=0)
miss_speed_sem = np.std(miss_speed, axis=0) / np.sqrt(len(miss_phases))

# Create the plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

# Plot 1: Vergence Angle
ax1.plot(time_grid, success_vergence_mean, '-', color=c_hunt, linewidth=2, label=f'Success')
ax1.fill_between(time_grid, 
                 success_vergence_mean - success_vergence_sem,
                 success_vergence_mean + success_vergence_sem,
                 alpha=0.3, color=c_hunt)

ax1.plot(time_grid, miss_vergence_mean, '-', color=c_non, linewidth=2, label=f'Miss')
ax1.fill_between(time_grid, 
                 miss_vergence_mean - miss_vergence_sem,
                 miss_vergence_mean + miss_vergence_sem,
                 alpha=0.3, color=c_non)

# Add phase boundaries
ax1.axvline(x=0, color='blue', linestyle='--', alpha=0.7)
ax1.axvline(x=1, color='orange', linestyle='--', alpha=0.7)

# # Add phase labels
# ax1.text(-0.5, ax1.get_ylim()[1] * 0.95, f'Pre-Detection\n({phase_duration} steps)', ha='center', fontsize=fontsize, 
#          bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
# ax1.text(0.5, ax1.get_ylim()[1] * 0.95, 'Tracking', ha='center', fontsize=fontsize,
#          bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
# ax1.text(1.5, ax1.get_ylim()[1] * 0.95, f'Post-Outcome\n({phase_duration} steps)', ha='center', fontsize=fontsize,
#          bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))

# ax1.set_xlabel('Normalized Time', fontsize=fontsize)
ax1.set_ylabel('Vergence Angle (°)', fontsize=fontsize)
# ax1.set_title('Vergence Angle Across Three Phases of Food Tracking')
ax1.legend(fontsize=fontsize)
ax1.grid(True, alpha=0.3)

# Plot 2: Speed
ax2.plot(time_grid, success_speed_mean, '-', color=c_hunt, linewidth=2, label=f'Success (N={len(success_phases)})')
ax2.fill_between(time_grid, 
                 success_speed_mean - success_speed_sem,
                 success_speed_mean + success_speed_sem,
                 alpha=0.3, color=c_hunt)

ax2.plot(time_grid, miss_speed_mean, '-', color=c_non, linewidth=2, label=f'Miss (N={len(miss_phases)})')
ax2.fill_between(time_grid, 
                 miss_speed_mean - miss_speed_sem,
                 miss_speed_mean + miss_speed_sem,
                 alpha=0.3, color=c_non)

# Add phase boundaries
ax2.axvline(x=0, color='blue', linestyle='--', alpha=0.7)
ax2.axvline(x=1, color='orange', linestyle='--', alpha=0.7)

# Add phase labels
ax2.text(-0.5, ax2.get_ylim()[1] * 0.95, f'Pre-Detection\n({phase_duration} steps)', ha='center', fontsize=fontsize)
         #bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
ax2.text(0.5, ax2.get_ylim()[1], 'Tracking', ha='center', fontsize=fontsize)
         #bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=1))
ax2.text(1.5, ax2.get_ylim()[1] * 0.95, f'Post-Outcome\n({phase_duration} steps)', ha='center', fontsize=fontsize)
         #bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))

ax2.set_xlabel('Normalized Time', fontsize=fontsize)
ax2.set_ylabel('Speed (mm/s)', fontsize=fontsize)
# ax2.set_title('Speed')
# ax2.legend(fontsize=fontsize)
ax2.grid(True, alpha=0.3)

fig.text(0.40, 0.53, "Detection", ha='center', va='center',
         color='blue', fontsize=fontsize)

fig.text(0.68, 0.53, "Outcome", ha='center', va='center',
         color='orange', fontsize=fontsize)

for ax in [ax1, ax2]:
    ax.tick_params(axis='both', which='major', labelsize=fontsize)  # bigger tick labels

sns.despine(ax=ax1)
sns.despine(ax=ax2)

plt.tight_layout()
# plt.savefig(f"{results_folder}/vergence_speed_three_phases.png", dpi=300)
plt.savefig(os.path.join(save_dir, "vergence_speed_three_phases.png"), bbox_inches="tight", dpi=300)
plt.show()


# In[ ]:




