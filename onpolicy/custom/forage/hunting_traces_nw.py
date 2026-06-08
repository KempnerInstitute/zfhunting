#!/usr/bin/env python
# coding: utf-8

# # Hunting traces
# 
# Code to create a figure similar to Figure 2G from [Johnson et al., 2020](https://www.cell.com/current-biology/pdf/S0960-9822(19)31465-4.pdf).

# In[ ]:


"""
# Run this as a CLI script
jupyter nbconvert hunting_traces_nw.ipynb --to python; python -u hunting_traces_nw.py 
"""

# !pip install seaborn


# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

from scipy.stats import linregress, ttest_rel, ttest_1samp


import utils_behavior as ub
from utils_behavior import calculate_polarization, calculate_cohesion, theil_index
from utils_features import add_hunting

from random import sample


import cfg
import os 
import glob

import sys
import argparse

import utils_report as ru

from utils_preprocess import get_df_with_candidate_vars

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

default_dir = "/srv/marl/raaghav/marl_zfish/rmappo-MultiAgentForagingEnv-1_agent/20250808_153214_1_bao_efp_0.05_vd_0.002_fd_10/outputs"
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


# In[ ]:


# Create a figure results folder
results_folder = f"{outputs_folder}/figures"
os.makedirs(results_folder, exist_ok=True)

print(f"Created folder: {results_folder}")


if not os.access(results_folder, os.W_OK):
    print("Cannot write to folder: results folder is /srv/marl/nathanwu/outputs/figures instead")

results_folder = "/home/nathanwu/outputs/test_outputs/figures"


# In[ ]:


#These are required for add_hunting()
# Convert vergence angle from radians to degrees for analysis
# Compute Vergence angle and speed
perception_field = 163 * np.pi / 180
dff['vergence_angle'] = dff['left_eye_angle'] - dff['right_eye_angle'] + perception_field
dff['vergence_angle_deg'] = dff['vergence_angle'] * 180 / np.pi
dff['speed'] = dff['displacement'] * cfg.ENV_PARAMS["fps_sim"]

tracking_sequences_df = ub.analyze_vergence_during_food_tracking(dff)

dff, _ = add_hunting(dff, list(dff.columns), tracking_sequences_df)

dff


# In[ ]:


eating_indices = dff.index[dff["eating_event"] == 1]
dff.loc[eating_indices | eating_indices - 1, ["food_positions", "food_ids", "eating_event", 'detected_food_ids', 'eaten_food_ids']]


# In[ ]:


dff = dff.sort_values(
    by=["env_id", "episode_index", "agent_id", "time_step"]
).reset_index(drop=True)

display(dff.head())


# In[ ]:


# dff["move_forward"] = dff["move_forward"] * cfg.FISH_CONSTANTS["max_speed"]
# dff["turn_angle"] = dff["turn_angle"] * cfg.FISH_CONSTANTS["max_turn_speed"]

assert (dff[f"displacement"] / dff["move_forward"])[1] > 0.124
assert (dff[f"displacement"] / dff["move_forward"])[1] < 0.126


# In[ ]:


dff['orientation'] = (dff['orientation'] + np.pi) % (2 * np.pi) - np.pi  # Normalize to [-pi, pi]


# In[ ]:


#First, flag all teleportation events :(
teleports = (dff[f"displacement"] / dff["move_forward"]) > 0.126

dff["teleport_event"] = teleports

dff.loc[(dff["teleport_event"] == True) | (dff["teleport_event"].shift(-1) == True), ["time_step", "position", "teleport_event"]]


# # By time steps

# In[ ]:


n_time_steps = 6


# In[ ]:


eating_df = pd.DataFrame(columns=list(dff.columns) + ["time_step_from_eating_event", "eating_event_no", "position_normalized"])

eating_indices = dff.index[dff["eating_event"] == 1]

i=0



#normalize position and orientation based on the first row
def rotate_and_normalize(p, origin=(0, 0), radians=0):
    R = np.array([[np.cos(radians), -np.sin(radians)],
                [np.sin(radians),  np.cos(radians)]])


    o = np.atleast_2d(origin)
    p = np.atleast_2d(p)

    return np.squeeze((R @ (p.T-o.T) + o.T).T) - o


for eating_idx in eating_indices:
    #make sure there are enough previous time steps
    if dff.loc[eating_idx, "time_step"] < n_time_steps:
        continue

    # Check for teleport events in the previous n_time_steps
    if dff.loc[eating_idx - n_time_steps:eating_idx, "teleport_event"].any():
        continue

    if not dff.loc[eating_idx - n_time_steps:eating_idx - 1, "hunting"].all():
        continue

    temp_df = dff.loc[eating_idx - n_time_steps:eating_idx, dff.columns].copy()

    temp_df["time_step_from_eating_event"] = temp_df["time_step"] - dff.loc[eating_idx, "time_step"]

    temp_df["eating_event_no"] = i
    i += 1
    
    
    positions = np.array(temp_df["position"].tolist())
    origin = np.array(temp_df.iloc[0]["position"])
    temp_df["position_normalized"] = list(rotate_and_normalize(positions, origin=origin, radians=-temp_df.iloc[0]["orientation"]))

    eating_df = pd.concat([eating_df, temp_df], ignore_index=True)

n_traces = len(eating_df["eating_event_no"].unique())

eating_df


# In[ ]:


#First, figure out which food was eaten
def get_food_eaten(group):
    try:
        food_eaten = list(set(group.iloc[-1]["eaten_food_ids"]) - set(group.iloc[-2]["eaten_food_ids"]))[0]
    except (IndexError, TypeError):
        food_eaten = np.nan
    group["food_eaten"] = food_eaten
    return group


def get_food_position(row):
    food_positions = row["food_positions"]
    food_ids = row["food_ids"]

    food_eaten = row["food_eaten"]

    try:
        idx = food_ids.index(food_eaten)
        return food_positions[idx]
    except ValueError:
        return np.nan


eating_df = eating_df.groupby("eating_event_no", group_keys=False).apply(get_food_eaten)
eating_df["eaten_food_position"] = eating_df.apply(get_food_position, axis=1)

eating_df["eaten_food_relative_position"] = eating_df["eaten_food_position"] - eating_df["position"]



eating_df["angle_to_food"] = eating_df.apply(lambda x: np.arctan2(x["eaten_food_relative_position"][1], x["eaten_food_relative_position"][0]), axis=1)

eating_df["relative_angle_to_food"] = (eating_df["angle_to_food"] - eating_df["orientation"]).apply(lambda x: (x + np.pi) % (2 * np.pi) - np.pi)

eating_df["postbout_relative_angle_to_food"] = eating_df.groupby("eating_event_no")["relative_angle_to_food"].shift(-1)

eating_df["distance_to_food"] = eating_df.apply(lambda x: np.linalg.norm(x["eaten_food_relative_position"]), axis=1)

eating_df["postbout_distance_to_food"] = eating_df.groupby("eating_event_no")["distance_to_food"].shift(-1)

eating_df["time_step_from_eating_event"]


# In[ ]:


from highlight_text import ax_text


# In[ ]:


fig = plt.figure(figsize = (3.6, 4.5))

gs = fig.add_gridspec(3, 2, height_ratios=[2.2,0, 1])

ax0 = fig.add_subplot(gs[0, :])
ax1 = fig.add_subplot(gs[2, 0])
ax2 = fig.add_subplot(gs[2, 1])

plt.sca(ax0)

red_labeled = False
blue_labeled = False


for i in sample(list(eating_df["eating_event_no"].unique()), 200):
    temp_df = eating_df[eating_df["eating_event_no"] == i]

    label = ""

    if temp_df.iloc[1]["turn_angle"] > 0:
        color = "red"
        if not red_labeled:
            label = "Left"
            red_labeled = True
    else:
        color = "blue"
        if not blue_labeled:
            label = "Right"
            blue_labeled = True
    
    plt.scatter(temp_df.iloc[-1]["position_normalized"][0], temp_df.iloc[-1]["position_normalized"][1], color=color, s=6, label= label, alpha = 0.5, zorder= 100)
    # plt.xlim(-5, 5)
    # plt.ylim(-5, 5)
    plt.xlabel("X Position (mm)")
    plt.ylabel("Y Position (mm)")
    plt.plot(temp_df["position_normalized"].apply(lambda x: x[0]), temp_df["position_normalized"].apply(lambda x: x[1]), linewidth = 1, alpha=0.1, color = "grey", label = "")
    plt.grid()

plt.arrow(-0.7, 0, 0.5, 0, head_width=0.1, head_length=0.2, fc='k', ec='k', zorder = 1000)

leg = plt.legend(title = "1st Turn\nDirection", fontsize = 8, markerscale=2, ncol = 1, bbox_to_anchor = (1, 0.5), loc='center left', handletextpad=0.2, columnspacing = 1)
for lh in leg.legendHandles:
    lh.set_alpha(1)

yabs_max = (np.max(np.abs(plt.ylim())))
plt.ylim(-yabs_max, yabs_max)

#plt.title(f"Last {n_time_steps} Steps of Successful Hunts")

plt.gca().set_aspect('equal', adjustable='box', anchor = 'W')

plt.sca(ax1)

# Fit linear regression to get slope and intercept
x = eating_df["relative_angle_to_food"]
y = eating_df["postbout_relative_angle_to_food"]
c = eating_df["distance_to_food"]

mask = ~np.isnan(x) & ~np.isnan(y)
slope, intercept, r_value, p_value, std_err = linregress(x[mask], y[mask])

if intercept >= 0:
    eqn_label = f"y = <{slope:.2f}>x + {intercept:.2f}"
else:
    eqn_label = f"y = <{slope:.2f}>x - {np.abs(intercept):.2f}"

#plt.plot(line_x_vals, slope*line_x_vals + intercept, color='orange', label=f"{eqn_label}, $r^2$ = {r_value**2:.2f}")

sns.regplot(x, y, label=f"{eqn_label}\n$r^2$={r_value**2:.2f}", ci=95, scatter_kws={"s": 0.4, "alpha": 0.05})

line_x_vals = np.array([np.min(x), np.max(x)])

# leg = plt.legend(markerscale = 10, fontsize=8, bbox_to_anchor = (0.5, 1.02), loc='lower center', handletextpad=0.2, columnspacing = 1)
# for lh in leg.legendHandles:
#     lh.set_alpha(1)

plt.plot(line_x_vals, line_x_vals, color='grey', linestyle='--', label='y=x Line')

#plt.title("Prey Azimuth (rad)", fontsize=10)

plt.xlabel("Prebout Prey\nAz. (rad)")
plt.ylabel("Postbout Prey\nAz. (rad)")

#plt.text(x = np.mean(plt.xlim()), y = plt.ylim()[1], s=f"{eqn_label}\n$r^2$={r_value**2:.2f}", ha="center", va="bottom", fontsize=8)

ax_text(x = np.mean(plt.xlim()), y = 0.2 * (plt.ylim()[1] - plt.ylim()[0]) + plt.ylim()[1], s=f"{eqn_label}", ha="center", va="bottom", fontsize=8, highlight_textprops=[{"color": "red", "weight": "bold"}])
plt.text(x = np.mean(plt.xlim()), y = plt.ylim()[1], s=f"\n$r^2$={r_value**2:.2f}", ha="center", va="bottom", fontsize=8)

#plt.title("Relative Angle to Food Before and After Bout \n(<5 time steps before eating)")

#plt.gca().set_aspect('equal', adjustable='box')
#plt.colorbar(label='Distance to Food', fraction=0.04, pad=0.04)


plt.sca(ax2)
x = eating_df["distance_to_food"]
y = eating_df["postbout_distance_to_food"]
mask = ~np.isnan(x) & ~np.isnan(y)
slope, intercept, r_value, p_value, std_err = linregress(x[mask], y[mask])
if intercept >= 0:
    eqn_label = f"y = <{slope:.2f}>x + {intercept:.2f}"
else:
    eqn_label = f"y = <{slope:.2f}>x - {np.abs(intercept):.2f}"

#plt.plot(line_x_vals, slope*line_x_vals + intercept, label=f"{eqn_label}, $r^2$ = {r_value**2:.2f}")
sns.regplot(x, y, label=f"{eqn_label}\n$r^2$={r_value**2:.2f}", ci=95, scatter_kws={"s": 0.4, "alpha": 0.05})

line_x_vals = np.array([np.min(x), np.max(x)])

# leg = plt.legend(markerscale = 10, fontsize=8, bbox_to_anchor = (0.5, 1.02), loc='lower center', handletextpad=0.2, columnspacing = 1)
# for lh in leg.legendHandles:
#     lh.set_alpha(1)


plt.plot(line_x_vals, line_x_vals, color='grey', linestyle='--', label='y=x Line')

#plt.title("Prey Distance (mm)", fontsize=10)

plt.xlabel("Prebout Prey\nDist. (mm)")
plt.ylabel("Postbout Prey\nDist. (mm)")
#plt.title("Distance to Food Before and After Bout\n(< 5 time steps before eating)")

# plt.gca().set_aspect('equal', adjustable='box')

ax_text(x = np.mean(plt.xlim()), y = 0.2 * (plt.ylim()[1] - plt.ylim()[0]) + plt.ylim()[1], s=f"{eqn_label}", ha="center", va="bottom", fontsize=8, highlight_textprops=[{"color": "red", "weight": "bold"}])
plt.text(x = np.mean(plt.xlim()), y = plt.ylim()[1], s=f"\n$r^2$={r_value**2:.2f}", ha="center", va="bottom", fontsize=8)

for ax in [ax0, ax1, ax2]:
    ax.tick_params(axis='both', direction='in')     
    sns.despine(ax = ax)

plt.tight_layout()

plt.subplots_adjust(wspace=1)

plt.savefig(f"{results_folder}/hunting_success_combo.png", dpi=300, bbox_inches = "tight")
plt.show()
print(len(x[mask]))


# # By hunting period

# In[ ]:


dff["hunting"]


# In[ ]:


tracking_sequences_df["end_time_step"] = tracking_sequences_df["start_time_step"] + tracking_sequences_df["tracking_duration"]
tracking_sequences_df


# In[ ]:


def get_row(dff, episode_index, time_step, env_id):
    return dff.loc[(dff["episode_index"] == episode_index) & (dff["time_step"] == time_step) & (dff["env_id"] == env_id)]


# In[ ]:


successful_tracking_sequences_df = tracking_sequences_df.loc[tracking_sequences_df["outcome"] == "success"]
successful_tracking_sequences_df


# In[ ]:


isolated_indices = []

for i in tracking_sequences_df.index:
    hunting_series = get_row(dff, 
                tracking_sequences_df.loc[i, "episode_index"], 
                tracking_sequences_df.loc[i, "start_time_step"] - 1, 
                tracking_sequences_df.loc[i, "env_id"]
                )["hunting"]
    if len(hunting_series) == 1 and hunting_series.values[0] == False:
        isolated_indices.append(i)

nonoverlapping_start_sequences_df = tracking_sequences_df.loc[isolated_indices]
nonoverlapping_start_sequences_df


# In[ ]:


n_steps = 8
step_range = np.arange(0, n_steps + 1)

cols = ["Event Index", "Event Type", "Time Step", "Period", "Vergence"]
vergence_events_df = pd.DataFrame(columns = cols)

#DETECTION
num_skipped = 0

for i in nonoverlapping_start_sequences_df.index:
    temp_df = pd.DataFrame(columns = cols)
    try:
        for time_step in step_range:
            vergence = get_row(dff, 
                            nonoverlapping_start_sequences_df.loc[i, "episode_index"], 
                            nonoverlapping_start_sequences_df.loc[i, "start_time_step"] + int(time_step), 
                            nonoverlapping_start_sequences_df.loc[i, "env_id"]
                            )["vergence_angle"]
            
            row = [i, "Detection", time_step, "Pre" if time_step < 0 else "Post", float(vergence)]

            temp_df.loc[len(temp_df)] = row

    except TypeError:
        num_skipped += 1
        continue

    vergence_events_df = pd.concat([vergence_events_df, temp_df])

print(f"{num_skipped} detection events skipped, {len(vergence_events_df.loc[vergence_events_df['Event Type'] == 'Detection']) // (n_steps)} detection events included")


# EATING
num_skipped = 0

for i in successful_tracking_sequences_df.index:
    temp_df = pd.DataFrame(columns = cols)
    try:
        for time_step in step_range:
            vergence = get_row(dff, 
                            successful_tracking_sequences_df.loc[i, "episode_index"], 
                            successful_tracking_sequences_df.loc[i, "end_time_step"] + int(time_step), 
                            successful_tracking_sequences_df.loc[i, "env_id"]
                            )["vergence_angle"]
            
            row = [i, "Eating", time_step, "Pre" if time_step < 0 else "Post", float(vergence)]

            temp_df.loc[len(temp_df)] = row

    except TypeError:
        num_skipped += 1
        continue

    vergence_events_df = pd.concat([vergence_events_df, temp_df])

print(f"{num_skipped} eating events skipped, {len(vergence_events_df.loc[vergence_events_df['Event Type'] == 'Eating']) // (n_steps)} eating events included")


# In[ ]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error


# In[ ]:


events_df = vergence_events_df.loc[vergence_events_df["Event Type"] == "Detection"]

X = events_df["Time Step"].values.reshape((-1, 1)).astype(float)
y = events_df["Vergence"].values.astype(float)

# Fit regression
model = LinearRegression().fit(X, y)
y_pred = model.predict(X)

# Compute stats
r2 = r2_score(y, y_pred)
#rmse = mean_squared_error(y, y_pred, squared=False)
coef = model.coef_[0]
intercept = model.intercept_

# Plot with seaborn
sns.regplot(x=X.flatten(), y=y, ci=95, line_kws={"color": "red"})

# Annotate equation, R², and RMSE
eq_text = f"$y = {coef:.2f}x + {intercept:.2f}$\n$R^2 = {r2:.3f}$"
plt.text(
    0.05, 0.95, eq_text,
    transform=plt.gca().transAxes,
    fontsize=12,
    verticalalignment="top",
    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7)
)

plt.show()

indices = np.unique(events_df["Event Index"])
for i in indices:
    sub_df = events_df.loc[events_df["Event Index"] == i]
    #print(len(sub_df))
    plt.plot(sub_df["Time Step"], sub_df["Vergence"], alpha=0.1)

plt.show()


# In[ ]:


vergence_events_df


# In[ ]:


vergence_slopes = []
for (event_idx, event_type), group in vergence_events_df.groupby(["Event Index", "Event Type"]):
    x = group["Time Step"].values.astype(float)/cfg.ENV_PARAMS["fps_sim"]
    y = group["Vergence"].values.astype(float)*(180/np.pi)

    slope, intercept, r_value, p_value, std_err = linregress(x, y)

    
    vergence_slopes.append({
        "Event Index": event_idx,
        "Event Type": event_type,
        "Slope": slope, #IN DEGREES PER SECOND
        "Intercept": intercept,
        "R2": r_value**2,
        "P Value": p_value,
        "-log(P Value)": - np.log10(p_value),
        "Std Err": std_err
    })

vergence_slopes_df = pd.DataFrame(vergence_slopes)
vergence_slopes_df


# In[ ]:


sns.violinplot(x="Event Type", y="Slope", data = vergence_slopes_df, order = ["Detection", "Eating"])

xlims = plt.xlim()

plt.hlines(0, xlims[0], xlims[1], "grey", linestyle = "--", zorder=100)

plt.xlim(xlims)

#sns.swarmplot(x="Event Type", y="Slope", data = vergence_slopes_df, s=5)


# In[ ]:


plt.figure(figsize = (2.3, 3.2))
# Your boxplot
ax = sns.boxplot(
    x="Event Type", y="Slope", data=vergence_slopes_df,
    order=["Detection", "Eating"],
    flierprops={"marker": "o", "markerfacecolor": "none", "markersize": 4},
    boxprops={"facecolor": "lightgrey"},
    showmeans=True,
    meanline=True,
    meanprops={'color': 'red'},
    palette="pastel",
    width = 0.5
)

# Draw horizontal reference line at y=0
xlims = plt.xlim()
plt.hlines(0, xlims[0], xlims[1], "k", zorder=0)
plt.xlim(xlims)

# --- Add n (sample size) ---
grouped = vergence_slopes_df.groupby("Event Type")["Slope"]
positions = {"Detection": 0, "Eating": 1}

for g, vals in grouped:
    n = len(vals)
    ax.text(
        positions[g],  # x position
        ax.get_ylim()[0] - 0.11*(ax.get_ylim()[1]-ax.get_ylim()[0]),  # slightly below axis
        f"n={n}", ha='center', va='top', fontsize=10
    )

# --- Significance test and stars ---
# Example: one-sample t-test vs 0 for each group
stars_pos = {}  # store y position to avoid overlap
for g, vals in grouped:
    t, p = ttest_1samp(vals, 0)
    if p < 0.001: stars = "***"
    elif p < 0.01: stars = "**"
    elif p < 0.05: stars = "*"
    else: stars = "ns"
    
    x = positions[g]
    y = max(vals) + 0.01*(ax.get_ylim()[1]-ax.get_ylim()[0])  # a bit above the box
    stars_pos[g] = y
    ax.text(x, y, stars, ha='center', va='bottom', fontsize=12, color="black")

# --- Add custom legend for mean line ---
mean_line = matplotlib.lines.Line2D([0], [0], linestyle='--', color='red', linewidth=1, label='Mean')

sns.despine()

ax.legend(handles=[mean_line], bbox_to_anchor = (1, 1.15), loc = "center right", handlelength = 1)

plt.tight_layout()

ax.tick_params(axis='both', direction='in') 


plt.ylabel("Rate of Vergence Change (°/s)")
plt.xlabel("\nEvent Type")

plt.savefig(f"{results_folder}/slope_of_vergence_change_boxes.png", dpi=300, bbox_inches = "tight")

plt.xlim(xlims)

for g, vals in grouped:
    mean_val = vals.mean()
    print(f"{g} mean = {mean_val:.2f}")


# In[ ]:


event_types = vergence_slopes_df["Event Type"].unique()
n_types = len(event_types)

fig, axes = plt.subplots(1, n_types, figsize=(6*n_types, 5), sharey=True)

if n_types == 1:
    axes = [axes]  # ensure iterable if only one event type

for ax, etype in zip(axes, event_types):
    subset = vergence_slopes_df[vergence_slopes_df["Event Type"] == etype]
    
    sns.scatterplot(
        data=subset,
        x="Slope",
        y="-log(P Value)",
        ax=ax,
        s=60,
    )
    
    # significance thresholds
    ax.axhline(-np.log10(0.05), color="red", linestyle="--", label="p=0.05")
    ax.axhline(-np.log10(0.01), color="blue", linestyle="--", label="p=0.01")

    ax.axvline(np.mean(subset["Slope"]), label = "Mean Slope")
    
    ax.set_title(f"Event Type: {etype}")

    
    #ax.set_yscale("log")  # log scale for p-values

    ax.invert_yaxis()
    

    plt.legend()

plt.tight_layout()

plt.savefig(f"{results_folder}/vergence_slope_p_vals.png", dpi=300, bbox_inches = "tight")
plt.show()


# In[ ]:



# assuming results_df contains the slopes from before
slope_tests = []

for event_type, group in vergence_slopes_df.groupby("Event Type"):
    slopes = group["Slope"].dropna().values

    t_stat, p_value = ttest_1samp(slopes, 0,)
    n = len(slopes)
    slope_tests.append({
        "Event Type": event_type,
        "N": n,
        "Mean Slope": np.mean(slopes),
        "T Stat": t_stat,
        "P Value": p_value/2, #1 sample
        "Std Error": np.std(slopes, ddof=1) / np.sqrt(n),
    })

slope_tests_df = pd.DataFrame(slope_tests)
slope_tests_df


# In[ ]:


# Convert p-value to stars 
def pval_to_stars(p): 
    if p < 0.001: 
        return "***" 
    elif p < 0.01: 
        return "**" 
    elif p < 0.05: 
        return "*" 
    else: 
        return "n.s."


# In[ ]:


# pick the smaller one-sided p-value for significance
slope_tests_df["P Value"].apply(pval_to_stars)

plt.figure(figsize=(3,3))

# bar plot with error bars
sns.barplot(
    data=slope_tests_df,
    x="Event Type",
    y="Mean Slope",
    yerr=slope_tests_df["Std Error"],
    capsize=0.2,
)

# add horizontal line at 0
plt.axhline(0, color="black", linewidth=1)

# add significance stars
for i, row in slope_tests_df.iterrows():
    x = i
    y = row["Mean Slope"]
    se = row["Std Error"]
    n = row["N"]
    star = pval_to_stars(row["P Value"])
    plt.text(x, y + se * (-1 if y < 0 else 1), star, ha="center", va=("bottom" if y >= 0 else "top"), fontsize=14, color="black", rotation = (180 if y < 0 else 0))

    plt.text(x, 0, f"\n\nn={n}", ha="center", va="bottom" if y < 0 else "center")

plt.ylim(np.array(plt.ylim()) * 1.2)

plt.ylabel("Mean Slope")
plt.tight_layout()
plt.savefig(f"{results_folder}/slope_of_vergence_change_bars.png", dpi=300, bbox_inches = "tight")
plt.show()


# In[ ]:


vergence_events_df_avgs = vergence_events_df.groupby(["Event Index", "Event Type", "Period"]).mean().reset_index()
vergence_events_df_avgs


# # Hunting period traces

# In[ ]:


min_hunting_length = 2


# In[ ]:


eating_df = pd.DataFrame(columns=list(dff.columns) + ["time_step_from_eating_event", "eating_event_no", "position_normalized"])

eating_indices = dff.index[dff["eating_event"] == 1]

i=0



#normalize position and orientation based on the first row
def rotate_and_normalize(p, origin=(0, 0), radians=0):
    R = np.array([[np.cos(radians), -np.sin(radians)],
                [np.sin(radians),  np.cos(radians)]])


    o = np.atleast_2d(origin)
    p = np.atleast_2d(p)

    return np.squeeze((R @ (p.T-o.T) + o.T).T) - o


for eating_idx in eating_indices:

    #Find the start of the hunting period

    hunting_start_idx = eating_idx - 1 
    
    while (
        hunting_start_idx > 0
        and dff.loc[hunting_start_idx, "time_step"] >= 0
        and dff.loc[hunting_start_idx, "hunting"]
        and not dff.loc[hunting_start_idx, "teleport_event"]
    ):
        hunting_start_idx -= 1

    hunting_start_idx += 1  # Move forward to the first index where hunting_status==1 and no teleport

    if eating_idx - hunting_start_idx < min_hunting_length:
        continue

    temp_df = dff.loc[hunting_start_idx:eating_idx, dff.columns].copy()

    temp_df["time_step_from_eating_event"] = temp_df["time_step"] - dff.loc[eating_idx, "time_step"]

    temp_df["eating_event_no"] = i
    i += 1
    
    
    positions = np.array(temp_df["position"].tolist())
    origin = np.array(temp_df.iloc[0]["position"])
    temp_df["position_normalized"] = list(rotate_and_normalize(positions, origin=origin, radians=-temp_df.iloc[0]["orientation"]))

    eating_df = pd.concat([eating_df, temp_df], ignore_index=True)


eating_df


# In[ ]:


plt.figure()
red_labeled = False
blue_labeled = False

n_traces = len(eating_df["eating_event_no"].unique())

for i in eating_df["eating_event_no"].unique():
    temp_df = eating_df[eating_df["eating_event_no"] == i]

    label = ""

    if temp_df.iloc[1]["turn_angle"] > 0:
        color = "red"
        if not red_labeled:
            label = "Right turn start"
            red_labeled = True
    else:
        color = "blue"
        if not blue_labeled:
            label = "Left turn start"
            blue_labeled = True
    
    plt.scatter(temp_df.iloc[-1]["position_normalized"][0], temp_df.iloc[-1]["position_normalized"][1], color=color, s=10, label= label)
    # plt.xlim(-5, 5)
    # plt.ylim(-5, 5)
    plt.xlabel("X Position (mm)")
    plt.ylabel("Y Position (mm)")
    plt.title(f"Fish Trajectory Leading to Eating Event from start of hunting (min time steps = {min_hunting_length}, {n_traces} traces)")
    if i==0:
        plt.arrow(-5, 0, 5, 0, head_width=0.5, head_length=1, fc='b', ec='b', zorder = 10000, label='Forward Direction' if i==0 else "")
    
    
    
    plt.plot(temp_df["position_normalized"].apply(lambda x: x[0]), temp_df["position_normalized"].apply(lambda x: x[1]), linewidth = 1, alpha=0.2, color = "grey", label = "")

    plt.grid()
    plt.gca().set_aspect('equal', adjustable='box')

plt.legend()

plt.savefig(f"{results_folder}/fish_eating_trajectory_hunting.png", dpi=300)

plt.show()

