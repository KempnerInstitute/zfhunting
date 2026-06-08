#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
# Run this as a CLI script
jupyter nbconvert bout_analysis_nw.ipynb --to python; python -u bout_analysis_nw.py 
"""

# !pip install seaborn


# In[ ]:


import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import gaussian_kde

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

from sklearn.cluster import KMeans, MiniBatchKMeans, SpectralClustering, AgglomerativeClustering

from sklearn.metrics import silhouette_score, davies_bouldin_score

from sklearn.decomposition import PCA

from utils_figstyle import *


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
#default_dir = "/srv/marl/nathanwu/rmappo-MultiAgentForagingEnv-check/20250921_174035_1_bao_vd_-0.006_lmp_-0.01_ltp_-0.01_fdr_10_run_1/outputs"
default_dir = "/srv/marl/nathanwu/rmappo-MultiAgentForagingEnv-check/20250922_213551_1_bao_vd_-0.006_lmp_-0.01_ltp_-0.01_fdr_10_run_6/outputs"

# #seed 0
# default_dir = "/srv/marl/nathanwu/rmappo-MultiAgentForagingEnv-check/20250924_150351_1_final_bao_vd_-0.006_lmp_-0.01_ltp_-0.01_fdr_10_wb_3_run_0/outputs"
# #seed 9
# default_dir = "/srv/marl/nathanwu/rmappo-MultiAgentForagingEnv-check/20250924_150413_1_final_bao_vd_-0.006_lmp_-0.01_ltp_-0.01_fdr_10_wb_3_run_9/outputs"
# #seed 8
# default_dir = "/srv/marl/nathanwu/rmappo-MultiAgentForagingEnv-check/20250924_150409_1_final_bao_vd_-0.006_lmp_-0.01_ltp_-0.01_fdr_10_wb_3_run_8/outputs"
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


dff = dff.sort_values(
    by=["env_id", "episode_index", "agent_id", "time_step"]
).reset_index(drop=True)

display(dff.head())


# In[ ]:


# dff["move_forward"] = dff["move_forward"] * cfg.FISH_CONSTANTS["max_speed"]
# dff["turn_angle"] = dff["turn_angle"] * cfg.FISH_CONSTANTS["max_turn_speed"]
dff['orientation'] = (dff['orientation'] + np.pi) % (2 * np.pi) - np.pi  # Normalize to [-pi, pi]


# # IF ONLY LOOKING AT HUNTING BEHAVIOR, MAKE ALL NON-HUNTING INTO NANS

# In[ ]:


#These are required for add_hunting()
# Convert vergence angle from radians to degrees for analysis
# Compute vergence angle and speed
perception_field = 163 * np.pi / 180
dff['vergence_angle'] = dff['left_eye_angle'] - dff['right_eye_angle'] + perception_field
dff['vergence_angle_deg'] = dff['vergence_angle'] * 180 / np.pi
dff['speed'] = dff['displacement'] * cfg.ENV_PARAMS["fps_sim"]

tracking_sequences_df = ub.analyze_vergence_during_food_tracking(dff)

dff, _ = add_hunting(dff, list(dff.columns), tracking_sequences_df)


print(f"{dff['hunting'].sum()/len(dff)} spent hunting")

#later on, we use nan displacements to flag things we don't want to analyze (teleports). We can use the same nan flagging here.
# dff.loc[dff["hunting"] == False, ["displacement", "move_forward", "turn_angle"]] = [np.nan] * 3
# dff[["displacement", "move_forward"]]


# # CONTINUE WITH OTHER STUFF

# In[ ]:


plt.scatter(dff["move_forward"], dff["turn_angle"], s=1, alpha=0.1)
plt.xlabel("forward speed (mm/s)")
plt.ylabel("turn speed (rad/s)")
plt.title("Move Forward vs Turn Angle")


# In[ ]:


dff["turn_angle_abs"] = np.abs(dff["turn_angle"])


# In[ ]:


x = dff["move_forward"].dropna()
y = dff["turn_angle"].dropna()

# Create JointGrid for shared axes
g = sns.JointGrid(x=x, y=y, space=0, height=2.5)

# Main 2D KDE
g.plot_joint(
    sns.kdeplot,
    cmap="Blues",
    shade=True,   # for older seaborn
    #bw=0.5        # older versions use 'bw', not 'bw_adjust'
)

# Marginals as histograms
#g.plot_marginals(sns.distplot, bins=30, color="k", kde=True)


# # Use matplotlib hist for marginals with spacing
g.ax_marg_x.hist(x, bins=30, edgecolor="black", color="lightgrey", alpha=1)  
g.ax_marg_y.hist(y, bins=30, edgecolor="black", color="lightgrey", alpha=1, orientation="horizontal")

# Labels + title
g.set_axis_labels("Forward Speed (mm/s)", "Turn Speed (rad/s)")
#plt.suptitle("Forward Speed vs Turn Speed", y=1.02)
plt.axis('tight') # Sets tight axis limits
plt.savefig(f"{results_folder}/Forward_speed_vs_turn_speed.png", dpi=300)
plt.show()


# In[ ]:


time_step = cfg.ENV_PARAMS["fps_sim"]
def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return np.vstack([x, y])

polar_coords = pd.DataFrame(columns = ["x", "y"])

rho_vals = dff["move_forward"].dropna()/time_step
phi_vals = dff["turn_angle"].dropna()/time_step

polar_coords["x"], polar_coords["y"] = pol2cart(rho_vals, phi_vals + np.pi/2)
polar_coords


plt.figure(figsize = (1.05, 2.5))

sns.kdeplot(polar_coords["x"], polar_coords["y"], cmap="Blues", shade=True, cut = 7)

#plt.scatter(polar_coords["x"], polar_coords["y"], alpha = 0.003)
plt.xlabel("X position (mm)")
plt.ylabel("Y position (mm)")

plt.gca().set_aspect('equal', adjustable="datalim")

# Forward arrow
plt.gca().arrow(0, -4.04, 0, 4, head_width=0.04, head_length=0.04,
             fc='k', ec='k', zorder=10000, label='Forward Direction')



plt.savefig(f"{results_folder}/movement_outcomes.png", dpi=300, bbox_inches="tight")


# # Movement Stats vs Distance to Food

# In[ ]:


def distance_to_closest_food(row):
    agent_position = np.array(row["position"])
    food_positions = np.array(row["food_positions"])
    distances = np.linalg.norm(food_positions - agent_position, axis=1)
    
    if len(distances) == 0:
        return np.nan
    return np.min(distances)
    

distances_to_food = dff.apply(distance_to_closest_food, axis=1)


# In[ ]:


def conditional_kde(x, y, gridsize=100, bandwidth=None):
    values = np.vstack([x, y])
    kde = gaussian_kde(values, bw_method=bandwidth)

    xi = np.linspace(x.min(), x.max(), gridsize)
    yi = np.linspace(y.min(), y.max(), gridsize)
    X, Y = np.meshgrid(xi, yi)

    coords = np.vstack([X.ravel(), Y.ravel()])
    Z = kde(coords).reshape(X.shape)

    # Normalize each vertical slice so that ∫ p(y|x) dy = 1
    Z /= Z.sum(axis=0, keepdims=True)
    return X, Y, Z


# In[ ]:


x = distances_to_food
y1 = dff["move_forward"]
y2 = dff["turn_angle"]

y2 = np.abs(y2)

y3 = dff["vergence_angle_deg"]

#y2 = np.abs(y2)

food_detection_range = cfg.AGENT_PARAMS["food_detection_range"]

x_lim = 2 * food_detection_range

# mask finite values only (no NaN, no inf)
mask = np.isfinite(x) & np.isfinite(y1) & np.isfinite(y2) & np.isfinite(y3) & (x <= x_lim)

x = x[mask]
y1 = y1[mask]
y2 = y2[mask]
y3 = y3[mask]


# In[ ]:


fig, axs = plt.subplots(3, 2, figsize=(5, 8), sharex="col", sharey="row")

#first subplot
plt.sca(axs[0, 0])
plt.title("Unnormalized")
sns.kdeplot(x, y1, shade=True, cmap="Blues", cut=0)
plt.ylabel("Movement speed (mm/s)")

plt.vlines([food_detection_range], plt.ylim()[0], plt.ylim()[1], "grey", linestyles="dashed")

plt.xlabel("")

#2nd subplot
plt.sca(axs[1, 0])
sns.kdeplot(x, y2, shade=True, cmap="Blues", cut=0)
plt.ylabel("Turn speed (rad/s)")

plt.vlines([food_detection_range], plt.ylim()[0], plt.ylim()[1], "grey", linestyles="dashed")

plt.xlabel("")

#3rd subplot
plt.sca(axs[2, 0])
sns.kdeplot(x, y3, shade=True, cmap="Blues", cut=0)
plt.ylabel("Vergence angle (°)")

plt.vlines([food_detection_range], plt.ylim()[0], plt.ylim()[1], "grey", linestyles="dashed")

plt.xlabel("Distance to food (mm)")
plt.xlim(0, x_lim)


# --- Subplot 1: Movement speed ---
plt.sca(axs[0, 1])

plt.title("Normalized")
X, Y, Z = conditional_kde(x, y1)
plt.pcolormesh(X, Y, Z, shading="auto", cmap="Blues")
plt.ylim(0, 5)
plt.vlines(food_detection_range, *plt.ylim(), "grey", linestyles="dashed", label="Food detection\nrange")
#plt.colorbar(label="Conditional density")
#plt.legend(loc="center left", bbox_to_anchor=(1.05, 0.5))

# --- Subplot 2: Turn speed ---
plt.sca(axs[1, 1])
X, Y, Z = conditional_kde(x, y2)
plt.pcolormesh(X, Y, Z, shading="auto", cmap="Blues")

plt.xlim(0, x_lim)
plt.ylim(0, 7)
plt.vlines(food_detection_range, *plt.ylim(), "grey", linestyles="dashed")


# --- Subplot 2: Turn speed ---
plt.sca(axs[2, 1])
X, Y, Z = conditional_kde(x, y3)
plt.pcolormesh(X, Y, Z, shading="auto", cmap="Blues")
plt.xlabel("Distance to food (mm)")
plt.xlim(0, x_lim)

plt.vlines(food_detection_range, *plt.ylim(), "grey", linestyles="dashed")
#plt.colorbar(label="Conditional density")

plt.suptitle("Behavioral Statistics by Distance to Food", y=1.03)
plt.tight_layout()

plt.savefig(f"{results_folder}/stats_vs_dist_to_food.png", dpi=300)
plt.show()


# In[ ]:


fig, axs = plt.subplots(1, 2, figsize=(4, 2.5), sharex=True)

#first subplot
plt.sca(axs[0])

sns.kdeplot(x[mask], y1[mask], shade=True, cmap="Blues", cut=0)
plt.ylabel("Forward Speed (mm/s)")
plt.xlabel("Distance to Food (mm)")

plt.vlines([food_detection_range], plt.ylim()[0], plt.ylim()[1], "grey", linestyles="dashed", label="Food Detection Radius")
#plt.legend()


#2nd subplot
plt.sca(axs[1])
sns.kdeplot(x[mask], y2[mask], shade=True, cmap="Blues", cut=0)
plt.ylabel("Turn Speed (rad/s)")

plt.vlines([food_detection_range], plt.ylim()[0], plt.ylim()[1], "grey", linestyles="dashed", label="Detection Radius")

plt.xlabel("Distance to Food (mm)")
plt.xlim(0, x_lim)

plt.legend(bbox_to_anchor=(1.00, 1.05), loc='lower right', borderaxespad=0.)
#plt.suptitle("Distribution of Movement/Turn Speeds by Distance to Food", y=1.03)

plt.tight_layout()

plt.savefig(f"{results_folder}/speeds_vs_dist_unnormalized.png", dpi=300, bbox_inches="tight")
plt.show()


# In[ ]:


plt.figure(figsize=(3, 3.2))
sns.kdeplot(x, y3, shade=True, cmap="Blues", cut=0)
plt.ylabel("Vergence angle (°)")

plt.vlines([food_detection_range], plt.ylim()[0], plt.ylim()[1], "grey", linestyles="dashed")

plt.xlabel("Distance to food (mm)")
plt.xlim(0, x_lim)

plt.vlines([food_detection_range], plt.ylim()[0], plt.ylim()[1], "grey", linestyles="dashed", label="Detection Radius")
plt.legend(bbox_to_anchor=(1.00, 1.05), loc='lower right', borderaxespad=0.)

plt.tight_layout()
plt.savefig(f"{results_folder}/vergence_vs_dist_unnormalized.png", dpi=300, bbox_inches="tight")

plt.show()


# # Dist to wall

# In[ ]:


x = dff["dist_to_wall"]
y1 = dff["move_forward"]
y2 = dff["turn_angle"]
y3 = dff["vergence_angle"]

y2 = np.abs(y2)

food_detection_range = cfg.AGENT_PARAMS["food_detection_range"]

x_lim = 2 * food_detection_range

# mask finite values only (no NaN, no inf)
mask = np.isfinite(x) & np.isfinite(y1) & np.isfinite(y2) & np.isfinite(y3) & (x <= x_lim)

x = x[mask]
y1 = y1[mask]
y2 = y2[mask]
y3 = y3[mask]


# In[ ]:


fig, axs = plt.subplots(3, 2, figsize=(5, 8), sharex="col", sharey="row")

#first subplot
plt.sca(axs[0, 0])
plt.title("Unnormalized")
sns.kdeplot(x, y1, shade=True, cmap="Blues", cut=0)
plt.ylabel("Movement speed (mm/s)")

plt.vlines([food_detection_range], plt.ylim()[0], plt.ylim()[1], "grey", linestyles="dashed")

plt.xlabel("")

#2nd subplot
plt.sca(axs[1, 0])
sns.kdeplot(x, y2, shade=True, cmap="Blues", cut=0)
plt.ylabel("Turn speed (rad/s)")

plt.vlines([food_detection_range], plt.ylim()[0], plt.ylim()[1], "grey", linestyles="dashed")

plt.xlabel("")

#3rd subplot
plt.sca(axs[2, 0])
sns.kdeplot(x, y3, shade=True, cmap="Blues", cut=0)
plt.ylabel("Vergence angle (rad)")

plt.vlines([food_detection_range], plt.ylim()[0], plt.ylim()[1], "grey", linestyles="dashed")

plt.xlabel("Distance to wall (mm)")
plt.xlim(0, x_lim)


# --- Subplot 1: Movement speed ---
plt.sca(axs[0, 1])

plt.title("Normalized")
X, Y, Z = conditional_kde(x, y1)
plt.pcolormesh(X, Y, Z, shading="auto", cmap="Blues")
plt.ylim(0, 5)
plt.vlines(food_detection_range, *plt.ylim(), "grey", linestyles="dashed", label="Food detection\nrange")
#plt.colorbar(label="Conditional density")
#plt.legend(loc="center left", bbox_to_anchor=(1.05, 0.5))

# --- Subplot 2: Turn speed ---
plt.sca(axs[1, 1])
X, Y, Z = conditional_kde(x, y2)
plt.pcolormesh(X, Y, Z, shading="auto", cmap="Blues")

plt.xlim(0, x_lim)
plt.ylim(0, 7)
plt.vlines(food_detection_range, *plt.ylim(), "grey", linestyles="dashed")


# --- Subplot 2: Turn speed ---
plt.sca(axs[2, 1])
X, Y, Z = conditional_kde(x, y3)
plt.pcolormesh(X, Y, Z, shading="auto", cmap="Blues")
plt.xlabel("Distance to wall (mm)")
plt.xlim(0, x_lim)

plt.vlines(food_detection_range, *plt.ylim(), "grey", linestyles="dashed")
#plt.colorbar(label="Conditional density")

plt.suptitle("Behavioral Statistics by Distance to Wall", y=1.03)
plt.tight_layout()

plt.savefig(f"{results_folder}/stats_vs_dist_to_wall.png", dpi=300)
plt.show()


# # Across n steps

# In[ ]:


dff


# In[ ]:


#Set displacement{n} to NaN where the ratio exceeds threshold. Basically get rid of teleporting times
mask = (dff[f"displacement"] / dff["move_forward"]) > 0.126

mask_indices = dff[mask].index
print(f"Teleports at {mask_indices}")

dff.loc[mask, f"move_forward"] = np.nan
dff.loc[mask, f"turn_angle"] = np.nan
dff.loc[mask, "displacement"] = np.nan


# In[ ]:


moves_and_turns = dff[["episode_index", "env_id", "time_step", "turn_angle", "turn_angle_abs", "position", "displacement", "move_forward"]].copy()

n_steps = 5
n_vals = range(1, n_steps)

moves_and_turns_grouped = moves_and_turns.groupby(["episode_index", "env_id"])

for n in n_vals:
    moves_and_turns[f"move_forward_{n}"] = moves_and_turns_grouped["move_forward"].shift(-n)
    moves_and_turns[f"turn_angle_{n}"] = moves_and_turns_grouped["turn_angle"].shift(-n)
    moves_and_turns[f"turn_angle_abs_{n}"] = np.abs(moves_and_turns[f"turn_angle_{n}"])


moves_and_turns[27700:27710]


# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Features and target
features = ["move_forward", "turn_angle"]

window_sizes = range(1, 16)
mse_results = []

for n in window_sizes:
    X_all, y_all = [], []
    
    for _, group in moves_and_turns_grouped:
        # Select relevant columns and drop NaN/inf rows
        group_clean = group[features].replace([np.inf, -np.inf], np.nan).dropna()
        values = group_clean.values
        
        if len(values) <= n:
            continue
        
        # Build sliding windows
        for i in range(len(values) - n):
            window = values[i:i+n]        # shape (n, 2)
            next_step = values[i+n]       # shape (2,)
            
            # Flatten the window into a feature vector of length 2*n
            X_all.append(window.flatten())
            y_all.append(next_step)
    
    if len(X_all) == 0:
        mse_results.append(np.nan)
        continue
    
    X_all = np.array(X_all, dtype=np.float64)
    y_all = np.array(y_all, dtype=np.float64)
    
    # Train linear regression
    model = LinearRegression()
    model.fit(X_all, y_all)
    preds = model.predict(X_all)
    
    # Compute mean squared error across both outputs
    mse = mean_squared_error(y_all, preds)
    mse_results.append(mse)

# Plot
plt.figure(figsize=(6,4))
plt.plot(window_sizes, mse_results, marker="o")
plt.xlabel("Window size (n)")
plt.ylabel("Mean Squared Error")
plt.title("MSE of predicting [move_forward, turn_angle] from n-step window")
plt.grid(True)

plt.savefig(f"{results_folder}/window_length.png", dpi=300)
plt.show()


# In[ ]:


# Exclude columns with 'turn_angle' in the name, unless they also contain '_abs'
pca_cols_full = moves_and_turns[[col for col in moves_and_turns.columns if "move_forward" in col or "turn_angle" in col and not "_abs" in col]]
pca_cols_abs_full = moves_and_turns[[col for col in moves_and_turns.columns if "move_forward" in col or "turn_angle" in col and "_abs" in col]]



pca_cols = pca_cols_full.dropna()
pca_cols_abs = pca_cols_abs_full.dropna()

pca_cols_abs


# In[ ]:


pca = PCA(n_components=3)
X_pca = pca.fit_transform(pca_cols.values)

print("Explained variance ratio:", pca.explained_variance_ratio_)
print("Total variance explained:", np.sum(pca.explained_variance_ratio_))

fig, axs = plt.subplots(1, 3, figsize=(18, 5))
col_name = "turn_angle"

colors = pca_cols[col_name]

scatter0 = axs[0].scatter(X_pca[:, 0], X_pca[:, 1], s=1, alpha=0.1, c=colors)
axs[0].set_xlabel(f"PCA 1\n(variance explained: {pca.explained_variance_ratio_[0]:.2f})")
axs[0].set_ylabel(f"PCA 2\n(variance explained: {pca.explained_variance_ratio_[1]:.2f})")
axs[0].set_title("PCA 1 vs 2")

scatter1 = axs[1].scatter(X_pca[:, 0], X_pca[:, 2], s=1, alpha=0.1, c=colors)
axs[1].set_xlabel(f"PCA 1\n(variance explained: {pca.explained_variance_ratio_[0]:.2f})")
axs[1].set_ylabel(f"PCA 3\n(variance explained: {pca.explained_variance_ratio_[2]:.2f})")
axs[1].set_title("PCA 1 vs 3")

scatter2 = axs[2].scatter(X_pca[:, 1], X_pca[:, 2], s=1, alpha=0.1, c=colors)
axs[2].set_xlabel(f"PCA 2\n(variance explained: {pca.explained_variance_ratio_[1]:.2f})")
axs[2].set_ylabel(f"PCA 3\n(variance explained: {pca.explained_variance_ratio_[2]:.2f})")
axs[2].set_title("PCA 2 vs 3")

plt.tight_layout()
cbar = fig.colorbar(scatter0, ax=axs, fraction=0.02, pad=0.04)
cbar.set_label("First turn velocity (rad/s)")

plt.suptitle(f"PCA on Move/Turn Speeds Over {n_steps} Steps", y=1.06)
plt.savefig(f"{results_folder}/move_turn_pca.png", dpi=300)
plt.show()


fig = plt.figure(figsize=(5, 5))
ax = plt.gca()
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], s=1, alpha=0.1, c=colors)
ax.set_xlabel(f"PCA 1\n(variance explained: {pca.explained_variance_ratio_[0]:.2f})")
ax.set_ylabel(f"PCA 2\n(variance explained: {pca.explained_variance_ratio_[1]:.2f})")
ax.set_title("Move/Turn PCA")
cbar = fig.colorbar(scatter, ax=ax, fraction=0.05, pad=0.04)
cbar.set_label("First turn velocity (rad/s)")
plt.savefig(f"{results_folder}/move_turn_pca_2d.png", dpi=300)
plt.show()


# In[ ]:


pca = PCA(n_components=3)
X_pca = pca.fit_transform(pca_cols_abs.values)

print("Explained variance ratio:", pca.explained_variance_ratio_)
print("Total variance explained:", np.sum(pca.explained_variance_ratio_))

fig, axs = plt.subplots(1, 3, figsize=(18, 5))
col_name = "turn_angle_abs"

colors = pca_cols_abs[col_name]

scatter0 = axs[0].scatter(X_pca[:, 0], X_pca[:, 1], s=1, alpha=0.1, c=colors)
axs[0].set_xlabel(f"PCA 1\n(variance explained: {pca.explained_variance_ratio_[0]:.2f})")
axs[0].set_ylabel(f"PCA 2\n(variance explained: {pca.explained_variance_ratio_[1]:.2f})")
axs[0].set_title("PCA 1 vs 2")

scatter1 = axs[1].scatter(X_pca[:, 0], X_pca[:, 2], s=1, alpha=0.1, c=colors)
axs[1].set_xlabel(f"PCA 1\n(variance explained: {pca.explained_variance_ratio_[0]:.2f})")
axs[1].set_ylabel(f"PCA 3\n(variance explained: {pca.explained_variance_ratio_[2]:.2f})")
axs[1].set_title("PCA 1 vs 3")

scatter2 = axs[2].scatter(X_pca[:, 1], X_pca[:, 2], s=1, alpha=0.1, c=colors)
axs[2].set_xlabel(f"PCA 2\n(variance explained: {pca.explained_variance_ratio_[1]:.2f})")
axs[2].set_ylabel(f"PCA 3\n(variance explained: {pca.explained_variance_ratio_[2]:.2f})")
axs[2].set_title("PCA 2 vs 3")

plt.tight_layout()
cbar = fig.colorbar(scatter0, ax=axs, fraction=0.02, pad=0.04)
cbar.set_label("First turn speed (rad/s)")

plt.suptitle(f"Move/Turn Magnitude PCA", y=1.06)
plt.savefig(f"{results_folder}/move_turn_abs_pca.png", dpi=300)
plt.show()


fig = plt.figure(figsize=(5, 5))
ax = plt.gca()
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], s=1, alpha=0.1, c=colors)
ax.set_xlabel(f"PCA 1\n(variance explained: {pca.explained_variance_ratio_[0]:.2f})")
ax.set_ylabel(f"PCA 2\n(variance explained: {pca.explained_variance_ratio_[1]:.2f})")
ax.set_title("Move/Turn Magnitude PCA")
cbar = fig.colorbar(scatter, ax=ax, fraction=0.05, pad=0.04)
cbar.set_label("First turn speed (rad/s)")
plt.savefig(f"{results_folder}/move_turn_abs_pca_2d.png", dpi=300)
plt.show()


# # Clustering

# In[ ]:


pca_cols


# In[ ]:


def cluster_eval(X, k_values = range(3, 10)):

    sil_scores = []

    for k in k_values:
        kmeans = KMeans(n_clusters=k) #can set randomstate
        labels = kmeans.fit_predict(X)
        print(f"kmeans {k} fit!")
        score = silhouette_score(X, labels, sample_size = 40000)
        sil_scores.append(score)
        print(f"k = {k}, silhouette score = {score:.3f}")

    # Find best k
    best_k = k_values[np.argmax(sil_scores)]
    print(f"\nBest number of clusters: {best_k}")

    # Plot silhouette scores
    plt.plot(k_values, sil_scores, marker='o')
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Silhouette score")
    plt.title("Silhouette Score vs k")

    plt.savefig(f"{results_folder}/silhouette_score.png", bbox_inches = "tight", dpi=300)
    plt.show()

    return best_k

best_k = cluster_eval(pca_cols)


# In[ ]:


kmeans = KMeans(n_clusters=best_k)
labels = kmeans.fit_predict(pca_cols) + 1

pca_cols


# In[ ]:


plt.figure(figsize = (18, 5))
for k in np.unique(labels):
    sub_df = pca_cols.loc[labels == k]
    plt.subplot(1, len(np.unique(labels)), k)
    plt.title(f"Cluster {k}")
    sns.kdeplot(sub_df["move_forward"], sub_df["turn_angle"], shade=True, cmap="Blues")
    plt.ylabel("First turn velocity (rad/s)")
    plt.xlabel("First forward velocity (mm/s)")

plt.tight_layout()


# In[ ]:


pca = PCA(n_components=3)

X_pca = pca.fit_transform(pca_cols.values)

fig = plt.figure(figsize=(1.5, 1.5))
ax = plt.gca()

# make sure labels are categorical (e.g. ints or strings)
unique_labels = np.unique(labels)

for lab in unique_labels:
    mask = labels == lab
    ax.scatter(
        X_pca[mask, 0],
        X_pca[mask, 1],
        s=0.05,
        alpha=0.1,
        label=f"{lab}"
    )

ax.set_xlabel(f"PC 1 ({pca.explained_variance_ratio_[0]*100:.0f}% var)")
ax.set_ylabel(f"PC 2 ({pca.explained_variance_ratio_[1]*100:.0f}% var)")
#ax.set_title(f"PCA on Move/Turn Speeds\nOver {n_steps} Steps")

leg = ax.legend(title="Cluster #", markerscale=20, fontsize=8, ncol = 1, bbox_to_anchor = (1.05, 0.5), loc='center left', handletextpad=0.2, columnspacing = 1)
for lh in leg.legendHandles:
    lh.set_alpha(1)

ax.tick_params(axis='both', direction='in') 
sns.despine()

plt.savefig(f"{results_folder}/move_turn_pca_clusters.png", dpi=300, bbox_inches = "tight")
plt.show()


# In[ ]:


#Fill back into the full df
# create a column of NaN in the original df
pca_cols_full["label"] = np.nan

# fill only the rows that were used (not null)
pca_cols_full.loc[pca_cols.index, "label"] = labels

pca_cols_full[115990:]


# In[ ]:


labels_full = pca_cols_full["label"].to_numpy()

# find valid consecutive transitions
valid_idx = (~pd.isna(labels_full[:-1])) & (~pd.isna(labels_full[1:]))

from_labels = labels_full[:-1][valid_idx].astype(int)
to_labels   = labels_full[1:][valid_idx].astype(int)

# build transition matrix (counts)
transition_df = pd.DataFrame({"from": from_labels, "to": to_labels})
transition_counts = pd.crosstab(transition_df["from"], transition_df["to"])

# normalize rows to get probabilities
transition_probs = transition_counts.div(transition_counts.sum(axis=1), axis=0)

print("Transition counts:\n", transition_counts)
print("\nTransition probabilities:\n", transition_probs)


# In[ ]:


plt.figure(figsize=(2.5, 2))

annot = np.where(transition_probs > 0.5,
                 transition_probs.round(1).astype(str),
                 "")
sns.heatmap(
    transition_probs,
    annot=annot,
    fmt="",
    cmap="Blues",
    vmin=0,
    vmax=1,
    annot_kws={"size": 9},
    cbar_kws={"ticks": [0, 1], "label": "Transition\nProbability"}
)
plt.gca().set_aspect("equal")

#plt.suptitle("Label Transition Probabilities", y=1.02)
plt.xlabel("To label")
plt.ylabel("From label")
plt.tight_layout()
plt.savefig(f"{results_folder}/label_transition_probabilities.png", dpi=300, bbox_inches = "tight")
plt.show()


# # Producing sample traces from each

# In[ ]:


num_traces = 20
n_steps = max(n_vals)


# In[ ]:


#normalize position and orientation based on the first row
def rotate_and_normalize(p, origin=(0, 0), radians=0):
    R = np.array([[np.cos(radians), -np.sin(radians)],
                [np.sin(radians),  np.cos(radians)]])


    o = np.atleast_2d(origin)
    p = np.atleast_2d(p)

    return np.squeeze((R @ (p.T-o.T) + o.T).T) - o


n_clusters = len(pca_cols_full["label"].dropna().unique())
n_cols = int(np.ceil(n_clusters))
n_rows = 1

fig, axes = plt.subplots(
    2, 2, figsize=(3, 4.5/3 * 2),
    sharex=True, sharey=True
)

axes = axes.flatten()  # convert 2D array → flat 1D array

# choose a categorical colormap (e.g. tab10, tab20, Set1)
#cmap = plt.cm.get_cmap("tab10", len(unique_labels))
default_colors = [c['color'] for c in plt.rcParams['axes.prop_cycle']]


for i, label in enumerate(sorted(pca_cols_full["label"].dropna().unique())):
    ax = axes[i]
    label_df = pca_cols_full.loc[pca_cols_full["label"] == label]

    indices = sample(list(label_df.index), num_traces)
    print(label, len(indices))

    for index in indices:
        orientations_and_angles = dff.loc[index:index+n_steps, ["orientation", "position"]].copy()

        positions = np.array(orientations_and_angles["position"].tolist())
        origin = np.array(orientations_and_angles.iloc[0]["position"])

        rotation = -orientations_and_angles.iloc[0]["orientation"]

        orientations_and_angles["position_normalized"] = list(
            rotate_and_normalize(positions, origin=origin, radians=rotation)
        )

        ax.scatter(
            orientations_and_angles.iloc[-1]["position_normalized"][0],
            orientations_and_angles.iloc[-1]["position_normalized"][1],
            color=default_colors[i], s=10, alpha=0.8, zorder=1000
        )

        ax.plot(
            orientations_and_angles["position_normalized"].apply(lambda x: x[0]),
            orientations_and_angles["position_normalized"].apply(lambda x: x[1]),
            linewidth=1, alpha=0.5, color="grey"
        )

    # Forward arrow
    ax.arrow(-1.03, 0, 1, 0, head_width=0.2, head_length=0.3,
             fc='k', ec='k', zorder=10000, label='Forward Direction')

    ax.set_title(f"Cluster {label:.0f}")
    ax.set_aspect('equal', adjustable='box')
    
    
    if i // 2 == 1:
        ax.set_xlabel("X (mm)")

    if i % 3 == 0:
        ax.set_ylabel("Y (mm)")

    ax.tick_params(axis='both', direction='in') 
    sns.despine()


# hide any unused subplots if grid > n_clusters
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])


plt.tight_layout()
plt.savefig(f"{results_folder}/trajectories_by_cluster.png", dpi=300)
plt.show()


# # Bout analysis when hunting and outside of hunting
# Just for one time step, not multiple

# In[ ]:


hunting_dff = dff.loc[dff["hunting"], ["move_forward", "turn_angle"]].copy().dropna()
hunting_dff


# In[ ]:


x = hunting_dff["move_forward"]
y = hunting_dff["turn_angle"]

# Create JointGrid for shared axes
g = sns.JointGrid(x=x, y=y, space=0, height=3)

# Main 2D KDE
g.plot_joint(
    sns.kdeplot,
    cmap="Blues",
    shade=True,   # for older seaborn
    #bw=0.5        # older versions use 'bw', not 'bw_adjust'
)

# Marginals as histograms
g.plot_marginals(sns.distplot, bins=30, color="gray", kde=False)

# Labels + title
g.set_axis_labels("forward speed (mm/s)", "turn speed (rad/s)")
plt.suptitle("Forward Speed vs Turn Speed During Hunting", y=1.02)

plt.savefig(f"{results_folder}/Forward_speed_vs_turn_speed_hunting.png", dpi=300)
plt.show()


# In[ ]:


best_k = cluster_eval(hunting_dff)


# In[ ]:


kmeans = KMeans(n_clusters=6)
labels = kmeans.fit_predict(hunting_dff)

hunting_dff


# In[ ]:


plt.figure()

unique_labels = np.unique(labels)
for lab in unique_labels:
    mask = labels == lab
    plt.scatter(
        hunting_dff.loc[mask, "move_forward"],
        hunting_dff.loc[mask, "turn_angle"],
        s=1,
        alpha=0.1,
        label=f"{lab}"
    )

plt.legend(title="Cluster number", markerscale=5, fontsize=8)

plt.xlabel("forward speed (mm/s)")
plt.ylabel("turn speed (rad/s)")

plt.savefig(f"{results_folder}/cluster_groups_hunting.png", dpi=300)

plt.show()


# In[ ]:


dff["displacement"].notna()


# In[ ]:


dff["hunting_labels"] = np.nan

dff.loc[hunting_dff.index, "hunting_labels"] = labels

dff["hunting_labels"].notna().sum()


# In[ ]:


labels_full = dff["hunting_labels"].to_numpy()

# find valid consecutive transitions
valid_idx = (~pd.isna(labels_full[:-1])) & (~pd.isna(labels_full[1:]))

from_labels = labels_full[:-1][valid_idx].astype(int)
to_labels   = labels_full[1:][valid_idx].astype(int)

# build transition matrix (counts)
transition_df = pd.DataFrame({"from": from_labels, "to": to_labels})
transition_counts = pd.crosstab(transition_df["from"], transition_df["to"])

# normalize rows to get probabilities
transition_probs = transition_counts.div(transition_counts.sum(axis=1), axis=0)

print("Transition counts:\n", transition_counts)
print("\nTransition probabilities:\n", transition_probs)

plt.figure(figsize=(6, 5))
sns.heatmap(
    transition_probs, 
    annot=True, fmt=".2f", cmap="Blues",
    cbar_kws={"label": "Transition Probability"}
)

plt.title("Label Transition Probability Matrix")
plt.xlabel("To label")
plt.ylabel("From label")
plt.tight_layout()
plt.savefig(f"{results_folder}/hunting_state_transition_probabilities.png", dpi=300)
plt.show()


# In[ ]:


dff.columns


# In[ ]:


# Get the indexes of successful hunts of length >= n
n_steps = 5


# Rolling check: create a boolean mask of where the last 5 rows are all hunting=True
hunting_n = dff["hunting"].rolling(window=n+1, min_periods=n+1).apply(lambda x: x.all()).fillna(0).astype(bool)

# Make sure there's no teleports
displacement_n = dff["displacement"].rolling(window=n+1, min_periods=n+1).apply(lambda x: x.notna().all()).fillna(0).astype(bool)

# Now require both: food eaten at this row, and 5 hunting before it
mask = dff["eating_event"] & hunting_n & displacement_n

all_successful_hunt_indices = dff.index[mask] - n

print(all_successful_hunt_indices)


# In[ ]:


dff.loc[all_successful_hunt_indices[0]:all_successful_hunt_indices[0] + n + 1, ["hunting", "eating_event", "hunting_labels"]]


# In[ ]:


num_traces = 10

fig = plt.figure()
ax = plt.gca()


indices = sample(list(all_successful_hunt_indices), num_traces)

# find unique hunting_labels
unique_labels = sorted(dff["hunting_labels"].dropna().unique())

# choose a categorical colormap (e.g. tab10, tab20, Set1)
#cmap = plt.cm.get_cmap("tab10", len(unique_labels))
default_colors = [c['color'] for c in plt.rcParams['axes.prop_cycle']]
cmap = matplotlib.colors.ListedColormap(default_colors)

# create a mapping from label -> color
norm = matplotlib.colors.BoundaryNorm(
    boundaries=np.arange(len(unique_labels)+1)-0.5,  # center bins on integers
    ncolors=len(unique_labels)
)

for index in indices:
    orientations_and_angles = dff.loc[index:index+n_steps, ["orientation", "position", "hunting_labels"]].copy()

    positions = np.array(orientations_and_angles["position"].tolist())
    origin = np.array(orientations_and_angles.iloc[0]["position"])

    rotation = -orientations_and_angles.iloc[0]["orientation"]

    orientations_and_angles["position_normalized"] = list(
        rotate_and_normalize(positions, origin=origin, radians=rotation)
    )

    ax.scatter(
        orientations_and_angles.iloc[-1]["position_normalized"][0],
        orientations_and_angles.iloc[-1]["position_normalized"][1],
        color="red", s=10, alpha=0.5, zorder=1000
    )

     # plot segment by segment
    for i in range(len(orientations_and_angles) - 1):
        x_vals = [
            orientations_and_angles.iloc[i]["position_normalized"][0],
            orientations_and_angles.iloc[i+1]["position_normalized"][0]
        ]
        y_vals = [
            orientations_and_angles.iloc[i]["position_normalized"][1],
            orientations_and_angles.iloc[i+1]["position_normalized"][1]
        ]

        label = orientations_and_angles.iloc[i]["hunting_labels"]

        color = cmap(norm(unique_labels.index(label)))

        ax.plot(x_vals, y_vals, color=color, linewidth=1, alpha=0.7)

# Forward arrow
ax.arrow(-0.5, 0, 0.5, 0, head_width=0.05, head_length=0.1,
            fc='b', ec='b', zorder=10000, label='Forward Direction')

ax.set_title(f"Hunting Trajectories (n = {num_traces})")
ax.set_aspect('equal', adjustable='box')
ax.set_xlabel("X Position (mm)")
ax.set_ylabel("Y Position (mm)")


plt.tight_layout()


# discrete colorbar
sm = matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, ticks=range(len(unique_labels)))
cbar.ax.set_yticklabels(unique_labels)
cbar.set_label("Hunting Label")


plt.show()

