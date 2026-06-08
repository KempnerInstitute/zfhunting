#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
# Run this as a CLI script
jupyter nbconvert report_rnn_decoding_sjy.ipynb --to python; python -u report_rnn_decoding_sjy.py 
"""

# !pip install seaborn


# In[ ]:


import os
import sys
import glob
import argparse

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from typing import Optional


import seaborn as sns
from scipy.stats import zscore

import sklearn
from sklearn.linear_model import Ridge, LinearRegression, LogisticRegression
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, f1_score
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import GroupKFold
# modular fns
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedShuffleSplit

import utils_report as ru

deg2rad = np.pi / 180.0


# In[ ]:


# Check if we're in interactive mode or batch mode
batchmode = False
if "ipykernel_launcher" in sys.argv[0]:
    print("Interactive mode")
else:
    batchmode = True
    print("Batch/CLI mode")


def get_latest_flat_pkl_file(input_dir="./"):
    pkl_files = glob.glob(input_dir + "/*.pkl")
    pkl_files = [f for f in pkl_files if "flat" in f]
    if not pkl_files:
        raise FileNotFoundError("No .pkl files found in the current directory.")
    latest_pkl_file = max(pkl_files, key=os.path.getctime)
    return latest_pkl_file

if batchmode:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "outputs_folder",
        nargs="?",
    )
    args = parser.parse_args()
    outputs_folder = args.outputs_folder
    latest_pkl = get_latest_flat_pkl_file(outputs_folder)
    dff = pd.read_pickle(latest_pkl)

else:
    dff = pd.read_pickle('/home/sonja/zfish/onpolicy/custom/forage/results/rmappo-MultiAgentForagingEnv-check/20250918_123636_1_sonja_for_decoding/outputs/MAZFish_neural_20250918_123636_sonja_for_decoding_agg_flattened.pkl')
    outputs_folder = '/home/sonja/zfish/onpolicy/custom/forage/results/rmappo-MultiAgentForagingEnv-check/20250918_123636_1_sonja_for_decoding/outputs/'


results_folder = f"{outputs_folder}/figures/"
os.makedirs(results_folder, exist_ok=True)


# In[ ]:


# ---------- utilities ----------
def _wrap_deg(x):
    # normalize to [-180, 180)
    return ((x + 180.0) % 360.0) - 180.0

def _angle_wrap_rad(x):
    # normalize to [-pi, pi)
    return (x + math.pi) % (2.0 * math.pi) - math.pi

def _bearing_and_distance(agent_pos, target_pos, heading_rad):
    vec = target_pos - agent_pos
    bearing_rad = math.atan2(vec[1], vec[0])           # world-frame bearing
    rel_rad = _angle_wrap_rad(bearing_rad - heading_rad)  # egocentric
    dist = float(np.linalg.norm(vec))
    return rel_rad, dist

# ---------- core generic helper ----------
def _distance_angle_for_ids(
    row,
    ids_key: str,
    *,
    out_prefix: str,
    orientation_in: str = "rad",        # "rad" or "deg"
    select: str = "nearest",            # "nearest" or "min_abs_angle"
    include_selected_id: bool = True
):
    """
    Compute egocentric angle (deg in [-180, 180)) and distance to a chosen target
    among IDs listed in `row[ids_key]`. Uses positions from food_ids/food_positions.
    """
    ids = row.get(ids_key, []) or []
    if not isinstance(ids, (list, tuple)) or len(ids) == 0:
        # empty -> NaNs
        base = {
            f"{out_prefix}_distance_to_food": np.nan,
            f"{out_prefix}_angle_to_food_deg": np.nan,
        }
        if include_selected_id:
            base[f"{out_prefix}_food_id_selected"] = np.nan
        return pd.Series(base)

    pos = np.asarray(row["position"], dtype=float)      # [x, y]
    heading = float(row["orientation"])
    if orientation_in == "deg":
        heading = math.radians(heading)                 # convert to radians

    food_ids = list(row["food_ids"])
    food_positions = [np.asarray(p, dtype=float) for p in row["food_positions"]]
    id2pos = {fid: p for fid, p in zip(food_ids, food_positions)}

    # candidates that actually exist in id2pos
    candidates = [(fid, id2pos[fid]) for fid in ids if fid in id2pos]
    if not candidates:
        base = {
            f"{out_prefix}_distance_to_food": np.nan,
            f"{out_prefix}_angle_to_food_deg": np.nan,
        }
        if include_selected_id:
            base[f"{out_prefix}_food_id_selected"] = np.nan
        return pd.Series(base)

    # compute (id, angle_rad, distance)
    computed = []
    for fid, tpos in candidates:
        rel_rad, dist = _bearing_and_distance(pos, tpos, heading)
        computed.append((fid, rel_rad, dist))

    if select == "min_abs_angle":
        target_id, rel_rad, dist = min(computed, key=lambda t: abs(t[1]))
    else:  # "nearest" (default)
        target_id, rel_rad, dist = min(computed, key=lambda t: t[2])

    rel_deg = _wrap_deg(math.degrees(rel_rad))

    out = {
        f"{out_prefix}_distance_to_food": float(dist),
        f"{out_prefix}_angle_to_food_deg": float(rel_deg),
    }
    if include_selected_id:
        out[f"{out_prefix}_food_id_selected"] = int(target_id)
    return pd.Series(out)

# ---------- convenience wrappers ----------
def add_distance_angle_for_ids(
    dff: pd.DataFrame,
    ids_key: str,
    *,
    out_prefix: str,
    count_col: Optional[str] = None,       # e.g., "num_binocular_food" (optional optimization)
    orientation_in: str = "rad",
    select: str = "nearest",
    include_selected_id: bool = True
):
    """
    Adds the columns:
      - f"{out_prefix}_distance_to_food"
      - f"{out_prefix}_angle_to_food_deg"
      - [optional] f"{out_prefix}_food_id_selected"
    computed from the id-list column `ids_key`.
    """
    if count_col is not None and count_col in dff:
        mask = dff[count_col].fillna(0).astype(int) > 0
    else:
        # fall back: compute where the list is non-empty
        mask = dff[ids_key].apply(lambda x: isinstance(x, (list, tuple)) and len(x) > 0)

    cols = [f"{out_prefix}_distance_to_food", f"{out_prefix}_angle_to_food_deg"]
    if include_selected_id:
        cols.append(f"{out_prefix}_food_id_selected")

    dff.loc[mask, cols] = dff.loc[mask].apply(
        lambda row: _distance_angle_for_ids(
            row,
            ids_key,
            out_prefix=out_prefix,
            orientation_in=orientation_in,
            select=select,
            include_selected_id=include_selected_id,
        ),
        axis=1,
    )

# ---------- example usage ----------
# 1) binocular
add_distance_angle_for_ids(
    dff,
    ids_key="binocular_food_ids",
    out_prefix="binoc",
    count_col="num_binocular_food",   # optional
    orientation_in="rad",
    select="nearest"                  # or "min_abs_angle"
)

# 2) monocular (build safely, then compute)
def _safe_set_diff(a, b):
    a = set(a or [])
    b = set(b or [])
    return list(a - b)

# if you have `detected_food_ids`, define monocular as detected - binocular:
dff["monocular_food_ids"] = dff.apply(
    lambda row: _safe_set_diff(row.get("detected_food_ids"), row.get("binocular_food_ids")),
    axis=1
)


add_distance_angle_for_ids(
    dff,
    ids_key="monocular_food_ids",
    out_prefix="monoc",
    orientation_in="rad",
    select="nearest"
)


# In[ ]:


# monocular_food_ids = detected food ids - binocular_food_ids (set subtraction)
dff["monocular_food_ids"] = dff.apply(
    lambda row: list(set(row["detected_food_ids"]) - set(row["binocular_food_ids"])), axis=1
)


# In[ ]:


import math
import numpy as np
import pandas as pd

# ---------- small helpers you already have ----------
def _wrap_deg(x):
    return ((x + 180.0) % 360.0) - 180.0

def _angle_wrap_rad(x):
    return (x + math.pi) % (2.0 * math.pi) - math.pi

# ---------- 1) Build a long-form "food tracks" table and get allocentric velocities ----------
def _build_food_tracks_with_velocity(dff: pd.DataFrame, *, seconds_per_step: float = 1.0):
    """
    Returns a DataFrame with columns:
      ['episode_index','env_id','time_step','food_id','food_x','food_y','vx','vy']
    vx, vy are allocentric (world-frame) velocities from finite differences.
    """
    req_cols = ["episode_index", "env_id", "time_step", "food_ids", "food_positions"]
    missing = [c for c in req_cols if c not in dff.columns]
    if missing:
        raise KeyError(f"Missing required columns in dff: {missing}")

    records = []
    # fast path: iterate once, no apply/explode overhead on nested lists
    epi = dff["episode_index"].to_numpy()
    env = dff["env_id"].to_numpy()
    t   = dff["time_step"].to_numpy()
    ids_series  = dff["food_ids"].to_list()
    poss_series = dff["food_positions"].to_list()

    for e, v, ts, ids, poss in zip(epi, env, t, ids_series, poss_series):
        if not isinstance(ids, (list, tuple)) or not isinstance(poss, (list, tuple)):
            continue
        for fid, pos in zip(ids, poss):
            if pos is None:
                continue
            records.append((e, v, ts, int(fid), float(pos[0]), float(pos[1])))

    ft = pd.DataFrame(records, columns=["episode_index","env_id","time_step","food_id","food_x","food_y"])
    if ft.empty:
        # Return an empty frame with expected columns
        ft["vx"] = np.nan
        ft["vy"] = np.nan
        return ft

    ft.sort_values(["episode_index","env_id","food_id","time_step"], inplace=True, kind="mergesort")

    ft["prev_food_x"]   = ft.groupby(["episode_index","env_id","food_id"])["food_x"].shift(1)
    ft["prev_food_y"]   = ft.groupby(["episode_index","env_id","food_id"])["food_y"].shift(1)
    ft["prev_time_step"]= ft.groupby(["episode_index","env_id","food_id"])["time_step"].shift(1)

    dt = (ft["time_step"] - ft["prev_time_step"]) * float(seconds_per_step)
    valid = dt > 0

    ft["vx"] = np.where(valid, (ft["food_x"] - ft["prev_food_x"]) / dt, np.nan)
    ft["vy"] = np.where(valid, (ft["food_y"] - ft["prev_food_y"]) / dt, np.nan)

    ft.drop(columns=["prev_food_x","prev_food_y","prev_time_step"], inplace=True)
    return ft

# ---------- 2) Merge selected food velocities and compute egocentric & LOS components ----------
def _merge_selected_food_velocity(
    dff: pd.DataFrame,
    ft: pd.DataFrame,
    *,
    prefix: str  # 'binoc' or 'monoc'
):
    sel_col = f"{prefix}_food_id_selected"
    if sel_col not in dff.columns:
        raise KeyError(f"Expected selected id column '{sel_col}' not found. Run add_distance_angle_for_ids(..., out_prefix='{prefix}', include_selected_id=True).")

    # rename ft columns to avoid collisions
    ft_ren = ft.rename(columns={
        "food_id": f"{prefix}_food_id",
        "food_x": f"{prefix}_food_x",
        "food_y": f"{prefix}_food_y",
        "vx":     f"{prefix}_food_vx",
        "vy":     f"{prefix}_food_vy",
    })

    dff = dff.merge(
        ft_ren[["episode_index","env_id","time_step", f"{prefix}_food_id", f"{prefix}_food_x", f"{prefix}_food_y", f"{prefix}_food_vx", f"{prefix}_food_vy"]],
        left_on=["episode_index","env_id","time_step", sel_col],
        right_on=["episode_index","env_id","time_step", f"{prefix}_food_id"],
        how="left"
    )

    # Compute body-frame forward/lateral components from allocentric vx, vy
    h = dff["orientation"].astype(float)  # radians
    vx = dff[f"{prefix}_food_vx"].astype(float)
    vy = dff[f"{prefix}_food_vy"].astype(float)

    dff[f"{prefix}_food_speed"]    = np.sqrt(vx*vx + vy*vy)
    dff[f"{prefix}_food_v_forward"] = vx*np.cos(h) + vy*np.sin(h)
    dff[f"{prefix}_food_v_lateral"] = -vx*np.sin(h) + vy*np.cos(h)  # +left

    # Ensure agent position_x / position_y exist
    if "position_x" not in dff.columns:
        dff["position_x"] = dff["position"].apply(lambda p: float(p[0]) if p is not None else np.nan)
    if "position_y" not in dff.columns:
        dff["position_y"] = dff["position"].apply(lambda p: float(p[1]) if p is not None else np.nan)

    # LOS (line-of-sight) radial/tangential components
    rx = dff[f"{prefix}_food_x"] - dff["position_x"]
    ry = dff[f"{prefix}_food_y"] - dff["position_y"]
    r  = np.sqrt(rx*rx + ry*ry)

    # Unit vectors (safe normalize)
    with np.errstate(invalid="ignore", divide="ignore"):
        r_hat_x = rx / r
        r_hat_y = ry / r
        t_hat_x = -r_hat_y  # left-handed perpendicular (counterclockwise)
        t_hat_y =  r_hat_x

        dff[f"{prefix}_food_v_radial"]     = vx*r_hat_x + vy*r_hat_y
        dff[f"{prefix}_food_v_tangential"] = vx*t_hat_x + vy*t_hat_y

    return dff

# ---------- 3) Bearing rate (deg/s) for the selected food when the id persists ----------
def _add_bearing_rate(dff: pd.DataFrame, *, seconds_per_step: float = 1.0, prefix: str):
    angle_col = f"{prefix}_angle_to_food_rad"
    sel_col   = f"{prefix}_food_id_selected"
    if angle_col not in dff.columns or sel_col not in dff.columns:
        # Angle or selected id missing; skip gracefully
        dff[f"{prefix}_bearing_rate_deg_per_s"] = np.nan
        return dff

    # group by episode/env to compute shifts in time
    grp_keys = ["episode_index","env_id"]
    prev_angle = dff.groupby(grp_keys)[angle_col].shift(1)
    prev_sel   = dff.groupby(grp_keys)[sel_col].shift(1)
    prev_t     = dff.groupby(grp_keys)["time_step"].shift(1)
    dt         = (dff["time_step"] - prev_t) * float(seconds_per_step)

    # only when the same selected id persists and dt>0
    same_id = (dff[sel_col].notna()) & (dff[sel_col] == prev_sel)
    valid   = same_id & dt.gt(0)

    dtheta = dff[angle_col] - prev_angle
    # wrap the difference into (-pi, pi]
    dtheta_wrapped = dtheta.apply(_angle_wrap_rad)

    rate_rad = np.where(valid, dtheta_wrapped / dt, np.nan)
    dff[f"{prefix}_bearing_rate_deg_per_s"] = np.degrees(rate_rad)
    return dff

# ---------- 4) Main convenience function ----------
def add_egocentric_food_motion(
    dff: pd.DataFrame,
    *,
    seconds_per_step: float = 1.0
):
    """
    Adds binocular/monocular motion features relative to the agent's egocentric frame and line-of-sight.
    Requires:
      - episode_index, env_id, time_step
      - position (2-vector), orientation (radians)
      - food_ids (list), food_positions (list of 2-vectors)
      - binoc_food_id_selected / monoc_food_id_selected (from your helper)
      - binoc_angle_to_food_rad / monoc_angle_to_food_rad (optional but used for bearing rate)
    """
    # Build allocentric velocities per food_id
    ft = _build_food_tracks_with_velocity(dff, seconds_per_step=seconds_per_step)

    # Merge for both variants and compute projections
    for prefix in ("binoc", "monoc"):
        dff = _merge_selected_food_velocity(dff, ft, prefix=prefix)
        dff = _add_bearing_rate(dff, seconds_per_step=seconds_per_step, prefix=prefix)

    return dff

# ---------- example usage ----------
# Ensure selected-id columns exist (you already do this with add_distance_angle_for_ids(..., include_selected_id=True))
# add_distance_angle_for_ids(dff, ids_key="binocular_food_ids", out_prefix="binoc", count_col="num_binocular_food", orientation_in="rad", select="nearest", include_selected_id=True)
# add_distance_angle_for_ids(dff, ids_key="monocular_food_ids", out_prefix="monoc", orientation_in="rad", select="nearest", include_selected_id=True)

# If your step duration is known, pass it (e.g., 0.05 for 20 Hz):
dff = add_egocentric_food_motion(dff, seconds_per_step=0.2)  # 5 Hz


# In[ ]:


# ---------- food movement utilities ----------
def _compute_food_movements(dff):
    """
    Compute allocentric movement (dx, dy) for each food item between consecutive timesteps.
    Returns a dict mapping (episode_index, env_id, timestep, food_id) -> (dx, dy)
    """
    food_movements = {}
    
    # Group by episode and environment
    for (episode_idx, env_id), group in dff.groupby(['episode_index', 'env_id']):
        group = group.sort_values('time_step').reset_index(drop=True)
        
        for i in range(1, len(group)):  # Start from time_step 1
            curr_row = group.iloc[i]
            prev_row = group.iloc[i-1]
            
            # Handle food_ids safely
            curr_food_ids = curr_row.get('food_ids', [])
            prev_food_ids = prev_row.get('food_ids', [])
            if curr_food_ids is None or (hasattr(curr_food_ids, '__len__') and len(curr_food_ids) == 0):
                curr_food_ids = []
            if prev_food_ids is None or (hasattr(prev_food_ids, '__len__') and len(prev_food_ids) == 0):
                prev_food_ids = []
            
            # Handle food_positions safely
            curr_food_positions = curr_row.get('food_positions', [])
            prev_food_positions = prev_row.get('food_positions', [])
            
            # Convert to list if it's an array/series, handle None/NaN
            if curr_food_positions is None or (isinstance(curr_food_positions, float) and np.isnan(curr_food_positions)):
                curr_food_positions = []
            elif hasattr(curr_food_positions, 'tolist'):  # numpy array or pandas series
                curr_food_positions = curr_food_positions.tolist()
            
            if prev_food_positions is None or (isinstance(prev_food_positions, float) and np.isnan(prev_food_positions)):
                prev_food_positions = []
            elif hasattr(prev_food_positions, 'tolist'):  # numpy array or pandas series  
                prev_food_positions = prev_food_positions.tolist()
            
            # Skip if we don't have valid data
            if not curr_food_ids or not prev_food_ids or not curr_food_positions or not prev_food_positions:
                continue
                
            # Create position lookup for previous time_step
            prev_id2pos = {}
            for fid, pos in zip(prev_food_ids, prev_food_positions):
                try:
                    prev_id2pos[fid] = np.asarray(pos, dtype=float)
                except (ValueError, TypeError):
                    continue  # Skip invalid positions
            
            # Compute movement for current time_step
            for fid, curr_pos in zip(curr_food_ids, curr_food_positions):
                if fid in prev_id2pos:
                    try:
                        curr_pos = np.asarray(curr_pos, dtype=float)
                        prev_pos = prev_id2pos[fid]
                        movement = curr_pos - prev_pos  # allocentric movement [dx, dy]
                        
                        key = (episode_idx, env_id, curr_row['time_step'], fid)
                        food_movements[key] = movement
                    except (ValueError, TypeError):
                        continue  # Skip invalid positions
    
    return food_movements

def _allocentric_to_egocentric_movement(allocentric_movement, agent_heading_rad):
    """
    Transform allocentric movement vector to egocentric coordinates.
    
    Args:
        allocentric_movement: [dx, dy] in world coordinates
        agent_heading_rad: agent's heading in radians
    
    Returns:
        [forward_speed, lateral_speed] in agent's reference frame
    """
    dx, dy = allocentric_movement
    cos_h = math.cos(agent_heading_rad)
    sin_h = math.sin(agent_heading_rad)
    
    # Rotate movement vector to agent's reference frame
    forward_speed = dx * cos_h + dy * sin_h    # positive = moving away from agent
    lateral_speed = -dx * sin_h + dy * cos_h   # positive = moving to agent's right
    
    return np.array([forward_speed, lateral_speed])

def _movement_stats_for_ids(
    row,
    ids_key: str,
    food_movements: dict,
    *,
    out_prefix: str,
    orientation_in: str = "rad",
    select: str = "nearest",  # "nearest", "min_abs_angle", or "fastest"
    include_selected_id: bool = True
):
    """
    Compute egocentric movement statistics for food items in the given ids list.
    """
    ids = row.get(ids_key, [])
    if ids is None or not isinstance(ids, (list, tuple)) or len(ids) == 0:
        base = {
            f"{out_prefix}_food_forward_speed": np.nan,
            f"{out_prefix}_food_lateral_speed": np.nan,
            f"{out_prefix}_food_speed": np.nan,
            f"{out_prefix}_food_movement_angle_deg": np.nan,
        }
        if include_selected_id:
            base[f"{out_prefix}_moving_food_id_selected"] = np.nan
        return pd.Series(base)

    episode_idx = row["episode_index"]
    env_id = row["env_id"] 
    time_step = row["time_step"]
    agent_pos = np.asarray(row["position"], dtype=float)
    agent_heading = float(row["orientation"])
    if orientation_in == "deg":
        agent_heading = math.radians(agent_heading)

    # Get food positions for distance calculations (for selection)
    food_ids = row.get("food_ids", [])
    food_positions = row.get("food_positions", [])
    
    # Handle potential None/NaN values
    if food_ids is None or food_positions is None:
        base = {
            f"{out_prefix}_food_forward_speed": np.nan,
            f"{out_prefix}_food_lateral_speed": np.nan,
            f"{out_prefix}_food_speed": np.nan,
            f"{out_prefix}_food_movement_angle_deg": np.nan,
        }
        if include_selected_id:
            base[f"{out_prefix}_moving_food_id_selected"] = np.nan
        return pd.Series(base)
    
    # Convert to lists if needed
    if hasattr(food_ids, 'tolist'):
        food_ids = food_ids.tolist()
    if hasattr(food_positions, 'tolist'):
        food_positions = food_positions.tolist()
    
    # Create position lookup
    id2pos = {}
    for fid, pos in zip(food_ids, food_positions):
        try:
            id2pos[fid] = np.asarray(pos, dtype=float)
        except (ValueError, TypeError):
            continue

    # Find candidates that have movement data and exist in current positions
    candidates = []
    for fid in ids:
        key = (episode_idx, env_id, time_step, fid)
        if key in food_movements and fid in id2pos:
            allocentric_movement = food_movements[key]
            egocentric_movement = _allocentric_to_egocentric_movement(allocentric_movement, agent_heading)
            forward_speed, lateral_speed = egocentric_movement
            total_speed = float(np.linalg.norm(egocentric_movement))
            
            # For selection, we might want distance to agent
            food_pos = id2pos[fid]
            distance = float(np.linalg.norm(food_pos - agent_pos))
            
            candidates.append((fid, forward_speed, lateral_speed, total_speed, distance))
    
    if not candidates:
        base = {
            f"{out_prefix}_food_forward_speed": np.nan,
            f"{out_prefix}_food_lateral_speed": np.nan,
            f"{out_prefix}_food_speed": np.nan,
            f"{out_prefix}_food_movement_angle_deg": np.nan,
        }
        if include_selected_id:
            base[f"{out_prefix}_moving_food_id_selected"] = np.nan
        return pd.Series(base)

    # Select based on criterion
    if select == "fastest":
        selected = max(candidates, key=lambda t: t[3])  # max total_speed
    elif select == "min_abs_angle":
        # For movement, we might select the one moving most directly toward/away from agent
        selected = min(candidates, key=lambda t: abs(t[2]))  # min abs(lateral_speed)
    else:  # "nearest" (default)
        selected = min(candidates, key=lambda t: t[4])  # min distance

    target_id, forward_speed, lateral_speed, total_speed, _ = selected
    
    # Compute movement angle in egocentric coordinates
    if total_speed > 0:
        movement_angle_rad = math.atan2(lateral_speed, forward_speed)
        movement_angle_deg = _wrap_deg(math.degrees(movement_angle_rad))
    else:
        movement_angle_deg = np.nan

    out = {
        f"{out_prefix}_food_forward_speed": float(forward_speed),
        f"{out_prefix}_food_lateral_speed": float(lateral_speed), 
        f"{out_prefix}_food_speed": float(total_speed),
        f"{out_prefix}_food_movement_angle_deg": float(movement_angle_deg),
    }
    if include_selected_id:
        out[f"{out_prefix}_moving_food_id_selected"] = int(target_id)
    return pd.Series(out)

def add_movement_stats_for_ids(
    dff: pd.DataFrame,
    ids_key: str,
    food_movements: dict,
    *,
    out_prefix: str,
    count_col: Optional[str] = None,
    orientation_in: str = "rad",
    select: str = "nearest",
    include_selected_id: bool = True
):
    """
    Adds movement statistics columns:
      - f"{out_prefix}_food_forward_speed"     # positive = moving away from agent
      - f"{out_prefix}_food_lateral_speed"     # positive = moving to agent's right  
      - f"{out_prefix}_food_speed"             # total movement speed
      - f"{out_prefix}_food_movement_angle_deg" # angle of movement in egocentric coords
      - [optional] f"{out_prefix}_moving_food_id_selected"
    """
    if count_col is not None and count_col in dff:
        mask = dff[count_col].fillna(0).astype(int) > 0
    else:
        mask = dff[ids_key].apply(lambda x: isinstance(x, (list, tuple)) and len(x) > 0)

    cols = [
        f"{out_prefix}_food_forward_speed",
        f"{out_prefix}_food_lateral_speed", 
        f"{out_prefix}_food_speed",
        f"{out_prefix}_food_movement_angle_deg"
    ]
    if include_selected_id:
        cols.append(f"{out_prefix}_moving_food_id_selected")

    # Initialize all columns with NaN
    for col in cols:
        dff[col] = np.nan

    if mask.any():
        dff.loc[mask, cols] = dff.loc[mask].apply(
            lambda row: _movement_stats_for_ids(
                row,
                ids_key,
                food_movements,
                out_prefix=out_prefix,
                orientation_in=orientation_in,
                select=select,
                include_selected_id=include_selected_id,
            ),
            axis=1,
        )

# ---------- usage ----------
# First compute all food movements
print("Computing food movements...")
food_movements = _compute_food_movements(dff)

# Add binocular food movement stats
add_movement_stats_for_ids(
    dff,
    ids_key="binocular_food_ids",
    food_movements=food_movements,
    out_prefix="binoc",
    count_col="num_binocular_food",
    orientation_in="rad",
    select="nearest"  # or "fastest", "min_abs_angle"
)

# Add monocular food movement stats  
add_movement_stats_for_ids(
    dff,
    ids_key="monocular_food_ids", 
    food_movements=food_movements,
    out_prefix="monoc",
    count_col="num_monocular_food",
    orientation_in="rad",
    select="nearest"
)

# Convert angles to radians for consistency
deg2rad = np.pi / 180.0
dff["binoc_food_movement_angle_rad"] = dff["binoc_food_movement_angle_deg"] * deg2rad
dff["monoc_food_movement_angle_rad"] = dff["monoc_food_movement_angle_deg"] * deg2rad

# # Update your column lists
# scalar_columns.extend([
#     "binoc_food_forward_speed",
#     "binoc_food_lateral_speed", 
#     "binoc_food_speed",
#     "monoc_food_forward_speed",
#     "monoc_food_lateral_speed",
#     "monoc_food_speed",
# ])

# circular_columns.extend([
#     "binoc_food_movement_angle_rad",
#     "monoc_food_movement_angle_rad",
# ])


# 

# In[ ]:


dff["position_x"] = dff["position"].apply(lambda pos: pos[0])
dff["position_y"] = dff["position"].apply(lambda pos: pos[1])
dff["num_detected_food"] = dff["detected_food_ids"].apply(lambda x: len(x) if isinstance(x, list) else 0)
dff["num_eaten_food"] = dff["eaten_food_ids"].apply(lambda x: len(x) if isinstance(x, list) else 0)
dff["num_nearby_food"] = dff["nearby_food_ids"].apply(lambda x: len(x) if isinstance(x, list) else 0)
dff["num_binocular_food"] = dff["binocular_food_ids"].apply(lambda x: len(x) if isinstance(x, list) else 0)
dff["num_monocular_food"] = dff["monocular_food_ids"].apply(lambda x: len(x) if isinstance(x, list) else 0)
dff["has_nearby"] = dff["num_nearby_food"] > 0
dff["orientation"] = dff["orientation"].apply(lambda x: (x + np.pi) % (2 * np.pi) - np.pi)  # Normalize to [-pi, pi]
dff["binoc_angle_to_food_rad"] = dff["binoc_angle_to_food_deg"] * deg2rad
dff["monoc_angle_to_food_rad"] = dff["monoc_angle_to_food_deg"] * deg2rad


scalar_columns = [
    "position_x",
    "position_y",
    "left_eye_angle",
    "right_eye_angle",
    "num_detected_food",
    "num_binocular_food",
    "num_eaten_food",
    "dist_to_wall",
    "vergence_angle",
    "binoc_distance_to_food",
    "monoc_distance_to_food",
    "num_monocular_food",
    "hunting"
]

circular_columns = [
    "orientation",
    "binoc_angle_to_food_rad",
    "monoc_angle_to_food_rad",
]


scalar_columns.extend([
    "binoc_food_forward_speed",
    "binoc_food_lateral_speed", 
    "binoc_food_speed",
    "monoc_food_forward_speed",
    "monoc_food_lateral_speed",
    "monoc_food_speed",
])

circular_columns.extend([
    "binoc_food_movement_angle_rad",
    "monoc_food_movement_angle_rad",
])


# In[ ]:


# scalar_columns += [
#     "binoc_food_vx", "binoc_food_vy", "binoc_food_speed",
#     "binoc_food_v_forward", "binoc_food_v_lateral",
#     "binoc_food_v_radial", "binoc_food_v_tangential",
#     "binoc_bearing_rate_deg_per_s",
#     "monoc_food_vx", "monoc_food_vy", "monoc_food_speed",
#     "monoc_food_v_forward", "monoc_food_v_lateral",
#     "monoc_food_v_radial", "monoc_food_v_tangential",
#     "monoc_bearing_rate_deg_per_s",
# ]


# In[ ]:


# add binoc_distance_to_food, binoc_angle_to_food (only for num_binocular_food > 0)
# dff


# In[ ]:


dff[dff['num_binocular_food'] > 0][['position', 'orientation', 'num_binocular_food', 'distance_to_food_hunting', 'angle_to_food_hunting_deg', 'hunting', 'binocular_food_ids', 'binoc_distance_to_food', 'binoc_angle_to_food_deg', 'monocular_food_ids', 'monoc_distance_to_food', 'monoc_angle_to_food_deg']]


# In[ ]:


# var = viz_columns[-1]
# dff[dff[var].notna()][var].hist(bins=50)
# plt.title(var)
# plt.ylabel("Density")


# In[ ]:


# simple_decoding.py
import numpy as np
import pandas as pd
from typing import Iterable, Optional, Union, Sequence, Tuple, Dict

from sklearn.linear_model import LinearRegression
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import KFold, GroupKFold
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt


# --------------------------
# Utilities
# --------------------------

def _ensure_2d(X: Union[np.ndarray, Sequence]) -> np.ndarray:
    """Make sure X is (n_samples, n_features). Accepts numpy arrays or list-like."""
    X = np.asarray(list(X)) if not isinstance(X, np.ndarray) else X
    if X.ndim == 1:
        X = X[:, None]
    return X

def _extract_X(
    data: Union[pd.DataFrame, np.ndarray, Sequence],
    X_col: Optional[str] = None
) -> np.ndarray:
    """
    Get the design matrix X.
    - If data is ndarray/sequence: treat as X directly.
    - If data is DataFrame: take column X_col (each cell containing 1D/2D arrays) and stack.
    """
    if isinstance(data, (np.ndarray, list, tuple)):
        return _ensure_2d(np.asarray(data))

    if isinstance(data, pd.DataFrame):
        if X_col is None:
            raise ValueError("When passing a DataFrame, please provide X_col (the column with vectors/embeddings).")
        # Accept vector in each cell (e.g., list/np.ndarray) or pre-expanded columns (2D array in each cell)
        X_list = data[X_col].tolist()
        X = np.asarray([np.asarray(x).ravel() for x in X_list])
        return _ensure_2d(X)

    raise TypeError("Unsupported data type for X.")

def _circular_to_components(y_rad: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Angle (radians) -> (cos, sin)."""
    return np.cos(y_rad), np.sin(y_rad)

def _circular_r2(y_true_cos, y_true_sin, y_pred_cos, y_pred_sin) -> float:
    """R^2 for 2D (cos, sin)."""
    y_true = np.column_stack([y_true_cos, y_true_sin])
    y_pred = np.column_stack([y_pred_cos, y_pred_sin])
    return r2_score(y_true, y_pred)

def _baseline_scalar_r2(y_train: np.ndarray, y_test: np.ndarray) -> float:
    """R^2 of mean baseline for scalar targets."""
    dummy = DummyRegressor(strategy="mean")
    dummy.fit(np.ones((len(y_train), 1)), y_train)
    yb = dummy.predict(np.ones((len(y_test), 1)))
    return r2_score(y_test, yb)

def _baseline_circular_r2(y_train_rad: np.ndarray, y_test_rad: np.ndarray) -> float:
    """R^2 baseline for circular using mean of cos and sin (separate constant predictors)."""
    ytr_c, ytr_s = _circular_to_components(y_train_rad)
    yte_c, yte_s = _circular_to_components(y_test_rad)

    dc = DummyRegressor(strategy="mean").fit(np.ones((len(ytr_c), 1)), ytr_c)
    ds = DummyRegressor(strategy="mean").fit(np.ones((len(ytr_s), 1)), ytr_s)

    yb_c = dc.predict(np.ones((len(yte_c), 1)))
    yb_s = ds.predict(np.ones((len(yte_s), 1)))
    return _circular_r2(yte_c, yte_s, yb_c, yb_s)


# --------------------------
# Core CV decoding
# --------------------------

def decode_scalar_cv(
    X: np.ndarray,
    y: np.ndarray,
    groups: Optional[Iterable] = None,
    n_splits: int = 5,
    min_train: int = 25,
    random_state: Optional[int] = 42,
) -> Dict[str, float]:
    """
    Linear decode a scalar target with K-fold/GroupKFold.
    Returns mean/std across folds for test R^2, baseline R^2, and normalized R^2.
    """
    X = _ensure_2d(X)
    y = np.asarray(y).ravel()
    if len(y) < max(min_train, n_splits):
        return {"perf_test_mean": np.nan, "perf_test_std": np.nan,
                "baseline_mean": np.nan, "baseline_std": np.nan,
                "norm_test_mean": np.nan, "norm_test_std": np.nan,
                "N_mean": len(y)}

    kf = GroupKFold(n_splits=n_splits) if groups is not None else KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    test_scores, base_scores, norm_scores, Ns = [], [], [], []
    for tr, te in kf.split(X, y, groups=groups):
        Xtr, Xte = X[tr], X[te]
        ytr, yte = y[tr], y[te]
        if len(np.unique(ytr)) < 2:  # degenerate folds
            continue

        model = LinearRegression()
        model.fit(Xtr, ytr)
        yhat = model.predict(Xte)

        r2 = r2_score(yte, yhat)
        r2b = _baseline_scalar_r2(ytr, yte)

        test_scores.append(r2)
        base_scores.append(r2b)
        norm_scores.append(r2 - r2b)
        Ns.append(len(te))

    if not test_scores:
        return {"perf_test_mean": np.nan, "perf_test_std": np.nan,
                "baseline_mean": np.nan, "baseline_std": np.nan,
                "norm_test_mean": np.nan, "norm_test_std": np.nan,
                "N_mean": 0}

    return {
        "perf_test_mean": float(np.mean(test_scores)),
        "perf_test_std": float(np.std(test_scores, ddof=1)) if len(test_scores) > 1 else 0.0,
        "baseline_mean": float(np.mean(base_scores)),
        "baseline_std": float(np.std(base_scores, ddof=1)) if len(base_scores) > 1 else 0.0,
        "norm_test_mean": float(np.mean(norm_scores)),
        "norm_test_std": float(np.std(norm_scores, ddof=1)) if len(norm_scores) > 1 else 0.0,
        "N_mean": int(np.mean(Ns)) if Ns else 0,
    }

def decode_circular_cv(
    X: np.ndarray,
    y_rad: np.ndarray,
    groups: Optional[Iterable] = None,
    n_splits: int = 5,
    min_train: int = 25,
    random_state: Optional[int] = 42,
) -> Dict[str, float]:
    """
    Linear decode a circular target (radians in [-pi, pi]) by predicting cos and sin.
    Aggregates to a single circular R^2.
    """
    X = _ensure_2d(X)
    y_rad = np.asarray(y_rad).ravel()
    if len(y_rad) < max(min_train, n_splits):
        return {"perf_test_mean": np.nan, "perf_test_std": np.nan,
                "baseline_mean": np.nan, "baseline_std": np.nan,
                "norm_test_mean": np.nan, "norm_test_std": np.nan,
                "N_mean": len(y_rad)}

    kf = GroupKFold(n_splits=n_splits) if groups is not None else KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    test_scores, base_scores, norm_scores, Ns = [], [], [], []
    for tr, te in kf.split(X, y_rad, groups=groups):
        Xtr, Xte = X[tr], X[te]
        ytr, yte = y_rad[tr], y_rad[te]

        ytr_c, ytr_s = _circular_to_components(ytr)
        yte_c, yte_s = _circular_to_components(yte)

        m_c = LinearRegression().fit(Xtr, ytr_c)
        m_s = LinearRegression().fit(Xtr, ytr_s)

        yhat_c = m_c.predict(Xte)
        yhat_s = m_s.predict(Xte)

        r2 = _circular_r2(yte_c, yte_s, yhat_c, yhat_s)
        r2b = _baseline_circular_r2(ytr, yte)

        test_scores.append(r2)
        base_scores.append(r2b)
        norm_scores.append(r2 - r2b)
        Ns.append(len(te))

    if not test_scores:
        return {"perf_test_mean": np.nan, "perf_test_std": np.nan,
                "baseline_mean": np.nan, "baseline_std": np.nan,
                "norm_test_mean": np.nan, "norm_test_std": np.nan,
                "N_mean": 0}

    return {
        "perf_test_mean": float(np.mean(test_scores)),
        "perf_test_std": float(np.std(test_scores, ddof=1)) if len(test_scores) > 1 else 0.0,
        "baseline_mean": float(np.mean(base_scores)),
        "baseline_std": float(np.std(base_scores, ddof=1)) if len(base_scores) > 1 else 0.0,
        "norm_test_mean": float(np.mean(norm_scores)),
        "norm_test_std": float(np.std(norm_scores, ddof=1)) if len(norm_scores) > 1 else 0.0,
        "N_mean": int(np.mean(Ns)) if Ns else 0,
    }


# --------------------------
# Multi-target runner
# --------------------------

# def decode_many(
#     data: Union[pd.DataFrame, np.ndarray, Sequence],
#     X_col: Optional[str],
#     targets_scalar: Iterable[str],
#     targets_circular: Iterable[str],
#     df_for_targets: Optional[pd.DataFrame] = None,
#     groups: Optional[Iterable] = None,
#     n_splits: int = 5,
#     min_train: int = 25,
#     random_state: Optional[int] = 42,
#     rename_map: Optional[Dict[str, str]] = None,
# ) -> pd.DataFrame:
#     """
#     Run CV decoding for many targets.
#     - data: either (n_samples, n_features) array, or a DataFrame plus X_col
#     - df_for_targets: where to pull y's from (defaults to `data` if it's a DataFrame)
#     - targets_*: column names in df_for_targets
#     Returns a tidy summary DataFrame with mean/std for each target.
#     """
#     X = _extract_X(data, X_col=X_col)
#     if df_for_targets is None:
#         if isinstance(data, pd.DataFrame):
#             df_for_targets = data
#         else:
#             raise ValueError("df_for_targets must be provided when data is not a DataFrame.")

#     results = []
#     for t in targets_scalar:
#         y = df_for_targets[t].to_numpy()
#         mask = ~np.isnan(y)
#         r = decode_scalar_cv(
#             X[mask], y[mask], None if groups is None else np.asarray(list(groups))[mask],
#             n_splits=n_splits, min_train=min_train, random_state=random_state
#         )
#         results.append({"target": t, "type": "scalar", **r})

#     for t in targets_circular:
#         # Expect radians in [-pi, pi]
#         y = df_for_targets[t].to_numpy()
#         mask = ~np.isnan(y)
#         r = decode_circular_cv(
#             X[mask], y[mask], None if groups is None else np.asarray(list(groups))[mask],
#             n_splits=n_splits, min_train=min_train, random_state=random_state
#         )
#         results.append({"target": t, "type": "circular", **r})

#     out = pd.DataFrame(results).sort_values("norm_test_mean", ascending=False).reset_index(drop=True)
#     if rename_map:
#         out["label"] = out["target"].map(lambda k: rename_map.get(k, k))
#     else:
#         out["label"] = out["target"]
#     return out

# # version that reuses the same rows across all targets
# def decode_many(
#     data: Union[pd.DataFrame, np.ndarray, Sequence],
#     X_col: Optional[str],
#     targets_scalar: Iterable[str],
#     targets_circular: Iterable[str],
#     df_for_targets: Optional[pd.DataFrame] = None,
#     groups: Optional[Iterable] = None,
#     n_splits: int = 5,
#     min_train: int = 25,
#     random_state: Optional[int] = 42,
#     rename_map: Optional[Dict[str, str]] = None,
#     equal_sample_binoc_monoc: bool = False,
# ) -> pd.DataFrame:
#     """
#     Run CV decoding for many targets.
#     - When equal_sample_binoc_monoc=True, downsample across rows to balance
#       binocular-present vs monocular-present rows (global), then intersect with
#       each target's valid mask.
#     """
#     X = _extract_X(data, X_col=X_col)
#     if df_for_targets is None:
#         if isinstance(data, pd.DataFrame):
#             df_for_targets = data
#         else:
#             raise ValueError("df_for_targets must be provided when data is not a DataFrame.")

#     rng = np.random.default_rng(random_state)
#     groups_arr = None if groups is None else np.asarray(groups)

#     # --- Global, across-rows balancing ---
#     def _present_any(df: pd.DataFrame, cols: list) -> np.ndarray:
#         arrs = []
#         for c in cols:
#             if c in df.columns:
#                 if c.startswith("num_"):
#                     arrs.append(df[c].fillna(0).to_numpy() > 0)
#                 else:
#                     arrs.append(df[c].notna().to_numpy())
#         return np.logical_or.reduce(arrs) if arrs else np.zeros(len(df), dtype=bool)

#     if equal_sample_binoc_monoc:
#         binoc_cols = ["num_binocular_food", "binoc_distance_to_food", "binoc_angle_to_food_rad"]
#         monoc_cols = ["num_monocular_food", "monoc_distance_to_food", "monoc_angle_to_food_rad"]

#         binoc_any = _present_any(df_for_targets, binoc_cols)
#         monoc_any = _present_any(df_for_targets, monoc_cols)
#         # Make mutually exclusive: rows with both count as binocular
#         monoc_any = np.logical_and(monoc_any, ~binoc_any)

#         binoc_idx = np.where(binoc_any)[0]
#         monoc_idx = np.where(monoc_any)[0]

#         if len(binoc_idx) > 0 and len(monoc_idx) > 0:
#             k = min(len(binoc_idx), len(monoc_idx))
#             b_sel = rng.choice(binoc_idx, size=k, replace=False)
#             m_sel = rng.choice(monoc_idx, size=k, replace=False)
#             sel_idx_global = np.sort(np.concatenate([b_sel, m_sel]))
#         else:
#             # Nothing to balance if one side is empty
#             sel_idx_global = np.arange(len(df_for_targets))
#     else:
#         sel_idx_global = np.arange(len(df_for_targets))

#     results = []

#     # --- Scalars ---
#     for t in targets_scalar:
#         y = df_for_targets[t].to_numpy()
#         valid = np.where(~np.isnan(y))[0]
#         sel_idx = np.intersect1d(valid, sel_idx_global, assume_unique=False)

#         r = decode_scalar_cv(
#             X[sel_idx],
#             y[sel_idx],
#             None if groups_arr is None else groups_arr[sel_idx],
#             n_splits=n_splits,
#             min_train=min_train,
#             random_state=random_state,
#         )
#         results.append({"target": t, "type": "scalar", **r})

#     # --- Circular (expects radians in [-pi, pi]) ---
#     for t in targets_circular:
#         y = df_for_targets[t].to_numpy()
#         valid = np.where(~np.isnan(y))[0]
#         sel_idx = np.intersect1d(valid, sel_idx_global, assume_unique=False)

#         r = decode_circular_cv(
#             X[sel_idx],
#             y[sel_idx],
#             None if groups_arr is None else groups_arr[sel_idx],
#             n_splits=n_splits,
#             min_train=min_train,
#             random_state=random_state,
#         )
#         results.append({"target": t, "type": "circular", **r})

#     out = pd.DataFrame(results).sort_values("norm_test_mean", ascending=False).reset_index(drop=True)
#     out["label"] = out["target"].map(lambda k: rename_map.get(k, k)) if rename_map else out["target"]
#     return out


# # --------------------------
# # Plotting
# # --------------------------

# def plot_importances(
#     summary_df: pd.DataFrame,
#     value_col: str = "norm_test_mean",
#     err_col: str = "norm_test_std",
#     title: str = "Decoding (normalized) performance",
#     xlabel: str = "Normalized test performance (model − baseline R²)",
#     outfile: Optional[str] = None,
#     xlim: Optional[Tuple[float, float]] = (0.0, 1.0),
#     type_colors: Optional[Dict[str, str]] = None,
# ) -> None:
#     """
#     Horizontal bars with error bars. Colors by target type (scalar/circular).
#     """
#     df = summary_df.copy()
#     df = df.sort_values(value_col, ascending=True)  # bottom->top small->large
#     labels = df.get("label", df["target"]).tolist()
#     y = np.arange(len(df))

#     if type_colors is None:
#         type_colors = {"scalar": "#1f77b4", "circular": "#ff7f0e"}

#     colors = [type_colors.get(t, "#7f7f7f") for t in df["type"]]

#     plt.figure(figsize=(8, max(4, 0.35 * len(df))))
#     plt.barh(y, df[value_col].values, xerr=df[err_col].values, color=colors, alpha=0.9)
#     plt.yticks(y, labels)
#     plt.xlabel(xlabel)
#     plt.title(title)
#     if xlim is not None:
#         plt.xlim(*xlim)
#     plt.gca().invert_yaxis()

#     # legend
#     uniq_types = list(dict.fromkeys(df["type"]))
#     handles = [plt.matplotlib.patches.Patch(color=type_colors.get(t, "#7f7f7f"), label=t) for t in uniq_types]
#     plt.legend(handles=handles, title="Target type", loc="lower right")

#     plt.tight_layout()
#     if outfile:
#         plt.savefig(outfile, dpi=300, bbox_inches="tight")
#     else:
#         plt.show()
#     plt.close()


# def decode_many(
#     data: Union[pd.DataFrame, np.ndarray, Sequence],
#     X_col: Optional[str],
#     targets_scalar: Iterable[str],
#     targets_circular: Iterable[str],
#     df_for_targets: Optional[pd.DataFrame] = None,
#     groups: Optional[Iterable] = None,
#     n_splits: int = 5,
#     min_train: int = 25,
#     random_state: Optional[int] = 42,
#     rename_map: Optional[Dict[str, str]] = None,
#     equal_sample_binoc_monoc: bool = False,
#     binoc_wins_ties: bool = True,   # ties (both present) count as binocular by default
# ) -> pd.DataFrame:
#     """
#     Run CV decoding for many targets.

#     If equal_sample_binoc_monoc=True:
#       - For EACH target separately, detect rows with any binocular vs any monocular signal
#         using the specified columns.
#       - Downsample the larger side by setting that target's y values to NaN (only for that target).
#       - Other targets are unaffected.
#     """
#     X = _extract_X(data, X_col=X_col)
#     if df_for_targets is None:
#         if isinstance(data, pd.DataFrame):
#             df_for_targets = data
#         else:
#             raise ValueError("df_for_targets must be provided when data is not a DataFrame.")

#     rng = np.random.default_rng(random_state)
#     groups_arr = None if groups is None else np.asarray(groups)

#     binoc_cols = ["num_binocular_food", "binoc_distance_to_food", "binoc_angle_to_food_rad"]
#     monoc_cols = ["num_monocular_food", "monoc_distance_to_food", "monoc_angle_to_food_rad"]

#     def _present_any(df: pd.DataFrame, cols: list) -> np.ndarray:
#         arrs = []
#         for c in cols:
#             if c in df.columns:
#                 if c.startswith("num_"):
#                     arrs.append(df[c].fillna(0).to_numpy() > 0)
#                 else:
#                     arrs.append(df[c].notna().to_numpy())
#         return np.logical_or.reduce(arrs) if arrs else np.zeros(len(df), dtype=bool)

#     # Precompute binocular/monocular presence (row-wise)
#     binoc_any = _present_any(df_for_targets, binoc_cols)
#     monoc_any = _present_any(df_for_targets, monoc_cols)

#     # Make mutually exclusive by default
#     if binoc_wins_ties:
#         monoc_any = np.logical_and(monoc_any, ~binoc_any)
#     else:
#         binoc_any = np.logical_and(binoc_any, ~monoc_any)

#     def _balance_with_nans(y: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
#         """Return a COPY of y with extra rows set to NaN to equalize binoc vs monoc within valid_mask."""
#         yb = y.astype(float).copy()  # ensure we can write NaNs (even for int/bool targets)
#         if not equal_sample_binoc_monoc:
#             return yb

#         b_idx = np.where(valid_mask & binoc_any)[0]
#         m_idx = np.where(valid_mask & monoc_any)[0]

#         if len(b_idx) == 0 or len(m_idx) == 0:
#             return yb  # nothing to balance

#         # Downsample the larger side to match the smaller
#         if len(b_idx) > len(m_idx):
#             drop = rng.choice(b_idx, size=len(b_idx) - len(m_idx), replace=False)
#         elif len(m_idx) > len(b_idx):
#             drop = rng.choice(m_idx, size=len(m_idx) - len(b_idx), replace=False)
#         else:
#             drop = np.array([], dtype=int)

#         if drop.size:
#             yb[drop] = np.nan
#         return yb

#     results = []

#     # Scalars
#     for t in targets_scalar:
#         y = df_for_targets[t].to_numpy()
#         valid = ~np.isnan(y.astype(float))
#         y_bal = _balance_with_nans(y, valid)
#         mask = ~np.isnan(y_bal)

#         r = decode_scalar_cv(
#             X[mask],
#             y_bal[mask],
#             None if groups_arr is None else groups_arr[mask],
#             n_splits=n_splits,
#             min_train=min_train,
#             random_state=random_state,
#         )
#         results.append({"target": t, "type": "scalar", **r})

#     # Circular (expects radians in [-pi, pi])
#     for t in targets_circular:
#         y = df_for_targets[t].to_numpy()
#         valid = ~np.isnan(y.astype(float))
#         y_bal = _balance_with_nans(y, valid)
#         mask = ~np.isnan(y_bal)

#         r = decode_circular_cv(
#             X[mask],
#             y_bal[mask],
#             None if groups_arr is None else groups_arr[mask],
#             n_splits=n_splits,
#             min_train=min_train,
#             random_state=random_state,
#         )
#         results.append({"target": t, "type": "circular", **r})

#     out = pd.DataFrame(results).sort_values("norm_test_mean", ascending=False).reset_index(drop=True)
#     out["label"] = out["target"].map(lambda k: rename_map.get(k, k)) if rename_map else out["target"]
#     return out

def decode_many(
    data: Union[pd.DataFrame, np.ndarray, Sequence],
    X_col: Optional[str],
    targets_scalar: Iterable[str],
    targets_circular: Iterable[str],
    df_for_targets: Optional[pd.DataFrame] = None,
    groups: Optional[Iterable] = None,
    n_splits: int = 5,
    min_train: int = 25,
    random_state: Optional[int] = 42,
    rename_map: Optional[Dict[str, str]] = None,
    equal_sample_binoc_monoc: bool = False,
    binoc_wins_ties: bool = True,     # ties (both present) => binocular by default
    nan_out_multi_food_hits: bool = False,  # NEW: NaN-out rows with multiple detections
) -> pd.DataFrame:
    """
    If nan_out_multi_food_hits=True:
      For EACH target, rows where num_binocular_food>1 or num_monocular_food>1
      are set to NaN for that target only.
    If equal_sample_binoc_monoc=True:
      For EACH target, downsample the larger (binoc vs monoc) side by setting
      that target's y to NaN (no global row drops).
    """
    X = _extract_X(data, X_col=X_col)
    if df_for_targets is None:
        if isinstance(data, pd.DataFrame):
            df_for_targets = data
        else:
            raise ValueError("df_for_targets must be provided when data is not a DataFrame.")

    rng = np.random.default_rng(random_state)
    groups_arr = None if groups is None else np.asarray(groups)

    # --- Presence helpers ---
    binoc_cols = ["num_binocular_food", "binoc_distance_to_food", "binoc_angle_to_food_rad"]
    monoc_cols = ["num_monocular_food", "monoc_distance_to_food", "monoc_angle_to_food_rad"]

    def _present_any(df: pd.DataFrame, cols: list) -> np.ndarray:
        arrs = []
        for c in cols:
            if c in df.columns:
                if c.startswith("num_"):
                    arrs.append(df[c].fillna(0).to_numpy() > 0)
                else:
                    arrs.append(df[c].notna().to_numpy())
        return np.logical_or.reduce(arrs) if arrs else np.zeros(len(df), dtype=bool)

    # Row-wise binocular/monocular presence (shared across targets)
    binoc_any = _present_any(df_for_targets, binoc_cols)
    monoc_any = _present_any(df_for_targets, monoc_cols)
    if binoc_wins_ties:
        monoc_any = np.logical_and(monoc_any, ~binoc_any)
    else:
        binoc_any = np.logical_and(binoc_any, ~monoc_any)

    # Multi-hit mask (shared across targets)
    num_binoc = df_for_targets.get("num_binocular_food", pd.Series(0, index=df_for_targets.index)).fillna(0).to_numpy()
    # If you actually meant a different column than num_monocular_food, change the next line:
    num_monoc = df_for_targets.get("num_monocular_food", pd.Series(0, index=df_for_targets.index)).fillna(0).to_numpy()
    multi_hits_mask = (num_binoc > 1) | (num_monoc > 1)

    def _balance_with_nans(y: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
        """
        Return a COPY of y with:
          1) multi-hit rows NaN'ed (if enabled),
          2) per-target binoc/monoc downsampling via NaNs (if enabled).
        """
        yb = y.astype(float).copy()  # ensure NaN-able
        # Step 1: multi-hit suppression
        if nan_out_multi_food_hits:
            drop_multi = np.where(valid_mask & multi_hits_mask)[0]
            if drop_multi.size:
                yb[drop_multi] = np.nan

        # Step 2: per-target balancing
        if equal_sample_binoc_monoc:
            # Effective valid after multi-hit NaNs
            eff_valid = valid_mask & ~multi_hits_mask if nan_out_multi_food_hits else valid_mask
            b_idx = np.where(eff_valid & binoc_any)[0]
            m_idx = np.where(eff_valid & monoc_any)[0]
            if b_idx.size and m_idx.size:
                if b_idx.size > m_idx.size:
                    drop = rng.choice(b_idx, size=b_idx.size - m_idx.size, replace=False)
                elif m_idx.size > b_idx.size:
                    drop = rng.choice(m_idx, size=m_idx.size - b_idx.size, replace=False)
                else:
                    drop = np.array([], dtype=int)
                if drop.size:
                    yb[drop] = np.nan
        return yb

    results = []

    # --- Scalars ---
    for t in targets_scalar:
        y = df_for_targets[t].to_numpy()
        valid = ~np.isnan(y.astype(float))
        y_bal = _balance_with_nans(y, valid)
        mask = ~np.isnan(y_bal)

        r = decode_scalar_cv(
            X[mask],
            y_bal[mask],
            None if groups_arr is None else groups_arr[mask],
            n_splits=n_splits,
            min_train=min_train,
            random_state=random_state,
        )
        results.append({"target": t, "type": "scalar", **r})

    # --- Circular (expects radians in [-pi, pi]) ---
    for t in targets_circular:
        y = df_for_targets[t].to_numpy()
        valid = ~np.isnan(y.astype(float))
        y_bal = _balance_with_nans(y, valid)
        mask = ~np.isnan(y_bal)

        r = decode_circular_cv(
            X[mask],
            y_bal[mask],
            None if groups_arr is None else groups_arr[mask],
            n_splits=n_splits,
            min_train=min_train,
            random_state=random_state,
        )
        results.append({"target": t, "type": "circular", **r})

    out = pd.DataFrame(results).sort_values("norm_test_mean", ascending=False).reset_index(drop=True)
    out["label"] = out["target"].map(lambda k: rename_map.get(k, k)) if rename_map else out["target"]
    return out


# In[ ]:


# 1) pick targets
scalar_targets = scalar_columns
# [
#     "position_x", "position_y", "num_detected_food", "num_binocular_food",
#     "num_eaten_food", "dist_to_wall", "move_forward", "turn_angle",
#     "displacement", "vergence_angle", "speed", "distance_to_food_hunting"
# ]
circular_targets = circular_columns

# 2) (optional) nicer labels for plotting
rename_map = {
    "position_x": "x position",
    "position_y": "y position",
    "num_detected_food": "# detected food",
    "num_binocular_food": "# binocular food",
    "num_eaten_food": "# eaten food",
    "dist_to_wall": "distance to wall",
    "move_forward": "forward move (mag)",
    "turn_angle": "turn angle (deg/s or rad/s)",
    "displacement": "displacement",
    "vergence_angle": "vergence angle",
    "speed": "speed",
    "distance_to_food_hunting": "distance to food (hunt)",
    "orientation": "orientation (rad)",
    "angle_to_food_hunting_deg": "angle to food (rad)"
}
rename_map = {}

# 3) run decoding (5-fold CV). If you have group splits (e.g., by episode), pass groups=...
summary = decode_many(
    data=dff,                       # you can also pass a 2D np.ndarray here
    X_col="rnn_states",             # column with vectors
    targets_scalar=scalar_targets,
    targets_circular=circular_targets,
    df_for_targets=dff,             # where y's come from
    groups=dff[["env_id","episode_index"]].astype(str).agg("_".join, axis=1),                    # or e.g., groups=dff[["env_id","episode_index"]].astype(str).agg("_".join, axis=1)
    n_splits=3,
    min_train=100,                  # raise if needed
    random_state=42,
    rename_map=rename_map,
    equal_sample_binoc_monoc=True,
    nan_out_multi_food_hits=True,   # NEW: NaN-out rows with multiple detections
)

# 4) plot importances (normalized performance with error bars)
plot_importances(
    summary_df=summary,
    title="Linear decoding (normalized R²) with error bars",
    outfile=results_folder + "decoding_importances_balance_no_multi.png",
    xlim=(0.0, 1.0)                 # adjust if your scores go outside this
)

print(summary.head(10))
summary.round(3).to_csv(results_folder + "decoding_summary_balance_no_multi.csv", index=False)


# In[ ]:





# In[ ]:




