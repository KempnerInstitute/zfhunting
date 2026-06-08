# NOTE: Util file but also accepts a flat_pkl as input and performs modular decoding analysis.

import os
import re
import traceback
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from pathlib import Path
import time
import argparse
import tqdm
from typing import Dict, List, Optional


import seaborn as sns
from scipy.stats import zscore

import sklearn
# from sklearn.model_selection import train_test_split, cross_val_score, KFold
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split
# from sklearn.inspection import permutation_importance
# from sklearn.svm import SVR, SVC
# from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
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

import statsmodels.api as sm

from utils_features import (
    add_knollen_by_dist,
    extract_knollen_direction_error_angle,
    add_morm_amp_field_features,
    add_distance_to_wall,
    add_distance_angle_to_closest_food,
    add_velocity_to_nearest_agent,
    add_eod_rolling_windows,
)
from cfg import FEATURE_METADATA, FEATURE_TYPE_COLORMAP, EXCLUDE_FROM_DECODING, AGENT_PARAMS, m_to_cm


#### FUNCTIONS FROM OLD VERSION USED IN MODULAR ####
def prepare_features_for_decoding(df, normalize=False, behavior_mode="homing"):
    """
    Prepare features for decoding analysis by adding them directly to the dataframe.

    Parameters:
    -----------
    df : pandas DataFrame
        The dataframe containing RNN states and features
    normalize : bool
        Whether to normalize features
    behavior_mode : str
        Whether to prepare features for 'homing' or 'foraging' task

    Returns:
    --------
    df : pandas DataFrame
        Dataframe with added features
    feature_names : list
        List of feature names added to the dataframe
    """
    df = df.copy()
    feature_names = []

    try:
        df, feature_names = add_knollen_by_dist(df, feature_names)
    except Exception as e:
        print("[ERROR]", e)
        traceback.print_exc()

    try:
        df, feature_names = extract_knollen_direction_error_angle(df, feature_names)
    except Exception as e:
        print("[ERROR]", e)

    # Add field info for sensors that sense magnitude and direction
    try:
        df, feature_names = add_morm_amp_field_features(df, feature_names)
    except Exception as e:
        print("[ERROR]", e)

    # Add distance to wall
    try:
        print("Adding distance to wall...")
        df, feature_names = add_distance_to_wall(df, feature_names)
    except Exception as e:
        print("[ERROR]", e)


    # NEW vars for foraging
    if behavior_mode == "foraging":
        # Add distance and angle to closest food
        try:
            print("Adding distance and angle to closest food...")
            df, feature_names = add_distance_angle_to_closest_food(df, feature_names)
        except Exception as e:
            print("[ERROR]", e)

        # Compute velocity of approach to nearest agent
        try:
            print("Adding velocity to nearest agent...")
            df, feature_names = add_velocity_to_nearest_agent(df, feature_names)
        except Exception as e:
            print("[ERROR]", e)

        try:
            print("Adding EOD rolling windows...")
            df, p_eod_features = add_eod_rolling_windows(df, eod_rate_window=11)
            feature_names.extend(p_eod_features)
        except Exception as e:
            print("[ERROR]", e)

        # Include other variables of interest that are already in the dataframe
        PREPROCESSED_FEATURES_TO_ADD = [
            "distance_to_nearest_agent",
            "size_of_nearest_agent",
            "distance_to_second_nearest_agent",
            "size_of_second_nearest_agent",
            "food_count_5cm",
            "food_front_5cm",
            "food_back_5cm",
            "food_left_5cm",
            "food_right_5cm",
        ]
        for existent_feature in PREPROCESSED_FEATURES_TO_ADD:
            if existent_feature in df.columns:
                feature_names.append(existent_feature)

    # Normalize features if requested
    if normalize:
        print("Normalizing features...")
        new_features = []
        original_features = list(feature_names)

        for feature_name in original_features:
            if not feature_name.endswith("_binned") and not feature_name.endswith(
                "_circular"
            ):
                try:
                    df[f"{feature_name}_normalized"] = (
                        df[feature_name] - df[feature_name].mean()
                    ) / df[feature_name].std()

                    new_features.append(f"{feature_name}_normalized")
                except Exception as e:
                    print(f"Could not normalize {feature_name}: {e}")

        feature_names.extend(new_features)

    return df, feature_names


def plot_regression_feature_distributions(
    df, regression_features, output_folder="./", bins=100
):
    """Plots histograms of all regression features as subplots."""
    regression_features = [f for f in regression_features if f in df.columns]
    num_features = len(regression_features)
    num_cols = 3  # Number of columns in subplot grid
    num_rows = -(-num_features // num_cols)  # Ceiling division for rows

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(4 * num_cols, 1.5 * num_rows))
    axes = axes.flatten()  # Flatten in case of single row

    for i, feature in tqdm.tqdm(enumerate(regression_features)):
        if "circular" in feature:
            continue
        ax = axes[i]
        # df[feature].hist(bins=bins, log=True, ax=ax)

        # Convert bool to int to avoid numpy.histogram crash
        value_series = df[feature]
        if value_series.dtype == bool or value_series.dtype == np.bool_:
            value_series = value_series.astype(int)
        if FEATURE_METADATA[feature].get("feature_type", "unknown") == "vector": # doesn't plot well
            continue
        # print(f"[INFO] Plotting feature: {feature} with {len(value_series)} values")
        value_series.hist(bins=bins, log=True, ax=ax)

        if "magnitude" in feature:
            ax.set_xscale("log")
        ax.set_title(feature, fontsize=10)

    # Hide any unused subplots
    for i in range(num_features, len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    fname = f"{output_folder}_regression_feature_distributions.png"
    plt.savefig(fname)
    print(f"[INFO] Saved regression feature distributions to {fname}")


########## DECODING REGRESSIONS MODULARIZED ##########
def stratified_downsample(df, downsample_nrows, random_state=42):
    """
    Return a uniformly random subset of exactly `downsample_nrows` rows,
    but sampled so that the relative frequency of each (env_id, episode_index, agent_id)
    stays roughly the same as in the full DataFrame.
    """
    if downsample_nrows is None or downsample_nrows >= len(df):
        return df

    # make a single-string “label” for each row's stratum
    strata = df[["env_id", "episode_index", "agent_id"]].astype(str) \
               .agg("_".join, axis=1)

    sss = StratifiedShuffleSplit(
        n_splits=1,
        test_size=downsample_nrows,
        random_state=random_state
    )

    # we only need the test‐indices from the split
    _, sampled_idx = next(sss.split(df, strata))
    return df.iloc[sampled_idx].reset_index(drop=True)


def extract_valid_data(
    train_df, test_df, feature_names, rnn_col="rnn_states", return_clean_df=False
):
    if isinstance(feature_names, str):
        feature_names = [feature_names]

    train_rows = train_df.shape[0]
    test_rows = test_df.shape[0]

    # Mask for rows with no NaNs in any of the requested features
    valid_train = ~train_df[feature_names].isna().any(axis=1)
    valid_test = ~test_df[feature_names].isna().any(axis=1)

    num_dropped_train = train_rows - valid_train.sum()
    num_dropped_test = test_rows - valid_test.sum()
    if num_dropped_train > 0 or num_dropped_test > 0:
        print(
            f" [WARN] Dropped {num_dropped_train} training rows and {num_dropped_test} test rows due to NaN values in features {feature_names}."
        )

    if return_clean_df:
        return train_df[valid_train].copy(), test_df[valid_test].copy()
    else:
        X_train = np.array(train_df[valid_train][rnn_col].tolist())
        X_test = np.array(test_df[valid_test][rnn_col].tolist())
        # y_train = train_df[valid_train][feature_names].values
        # y_test = test_df[valid_test][feature_names].values
        if len(feature_names) == 1:
            y_train = train_df[valid_train][feature_names[0]].values
            y_test = test_df[valid_test][feature_names[0]].values
        else:
            y_train = train_df[valid_train][feature_names].values
            y_test = test_df[valid_test][feature_names].values
        return X_train, y_train, X_test, y_test


def get_regression_features(df, feature_names):
    regression_features = [ k for k, v in FEATURE_METADATA.items() if v.get("feature_type", "unknown") in ["scalar", "vector", "circular", "probability", "boolean", "count"]]
    regression_features += feature_names # from prepare_features_for_decoding() above
    regression_features = list(set(regression_features))  # Remove duplicates

    not_in_feature_metadata = set(regression_features) - set(FEATURE_METADATA.keys())
    if not_in_feature_metadata:
        print(f"Warning: The following features are not in FEATURE_METADATA: {not_in_feature_metadata}")

    regression_features = [
        f for f in regression_features if f in df.columns if f not in (EXCLUDE_FROM_DECODING + list(not_in_feature_metadata))
    ]

    return regression_features


def get_sensor_ranges(metadata, behavior_mode):
    """
    Extract detection ranges from metadata or cfg defaults, assemble
    sensor_ranges: list of ((min, max), label, abbrev, distance_col)
    """
    agent_args = metadata.get('agent_args', {})

    # Convert to cm
    morm_agent_cm = agent_args.get(
        'morm_agent_detection_range', AGENT_PARAMS['morm_agent_detection_range']
    ) * m_to_cm
    knollen_agent_cm = agent_args.get(
        'knollen_agent_detection_range', AGENT_PARAMS['knollen_agent_detection_range']
    ) * m_to_cm
    morm_food_cm = agent_args.get(
        'morm_food_detection_range', AGENT_PARAMS.get('morm_food_detection_range', AGENT_PARAMS['morm_agent_detection_range'])
    ) * m_to_cm

    sensor_ranges = []
    # Agent-based ranges, criterion 'distance_to_nearest_agent'
    sensor_ranges.append(((-np.inf, np.inf), 'Unrestricted', 'unrestricted_agent', 'distance_to_nearest_agent'))
    sensor_ranges.append(((0, morm_agent_cm), f'Mormyromast range ({morm_agent_cm:.1f}cm)', 'morm_agent', 'distance_to_nearest_agent'))
    sensor_ranges.append(((morm_agent_cm, knollen_agent_cm), f'Knollen range ({knollen_agent_cm:.1f}cm)', 'knollen_agent', 'distance_to_nearest_agent'))
    sensor_ranges.append(((knollen_agent_cm, np.inf), f'Outside sensing (> {knollen_agent_cm:.1f}cm)', 'outside_agent', 'distance_to_nearest_agent'))

    if behavior_mode == 'foraging':
        # Food-based ranges, criterion 'distance_to_closest_food'
        sensor_ranges.append(((0, morm_food_cm), f'Mormyromast food range ({morm_food_cm:.1f}cm)', 'morm_food', 'distance_to_closest_food'))
        sensor_ranges.append(((morm_food_cm, np.inf), f'Outside food (> {morm_food_cm:.1f}cm)', 'outside_food', 'distance_to_closest_food'))

    return sensor_ranges


###### BASELINE R2 CALCULATIONS ######
# from older decoding version but called in modular
def circular_r2_score(y_true_cos, y_true_sin, y_pred_cos, y_pred_sin):
    """
    Calculate R2 score for circular predictions based on cos and sin components.

    Parameters:
    -----------
    y_true_cos : array-like
        True cosine values
    y_true_sin : array-like
        True sine values
    y_pred_cos : array-like
        Predicted cosine values
    y_pred_sin : array-like
        Predicted sine values

    Returns:
    --------
    float
        R2 score for the circular prediction
    """
    # Sanity check: all arrays must have the same length
    lengths = [len(y_true_cos), len(y_true_sin), len(y_pred_cos), len(y_pred_sin)]
    if len(set(lengths)) > 1:
        print("[ERROR] Length mismatch in circular_r2_score:")
        print(f"  y_true_cos: {len(y_true_cos)}")
        print(f"  y_true_sin: {len(y_true_sin)}")
        print(f"  y_pred_cos: {len(y_pred_cos)}")
        print(f"  y_pred_sin: {len(y_pred_sin)}")
        raise ValueError(
            f"Found input variables with inconsistent numbers of samples: {lengths}"
        )

    # Stack the true and predicted values as 2D arrays
    y_true = np.column_stack((y_true_cos, y_true_sin))
    y_pred = np.column_stack((y_pred_cos, y_pred_sin))

    # Calculate R2 score for 2D data
    return r2_score(y_true, y_pred)


def calculate_baseline_r2_scalar(y_train, y_test, y_pred=None, baseline_type="mean", n_shuffles=30, random_state=None,):
    rng = np.random.RandomState(random_state)
    
    if baseline_type == "shuffle":
        if y_pred is None:
            raise ValueError("y_pred must be provided for shuffle baseline.")
        scores = []
        for _ in range(n_shuffles):
            shuffled = y_pred.copy()
            rng.shuffle(shuffled)
            scores.append(r2_score(y_test, shuffled))
        return np.mean(scores)

    elif baseline_type in ["mean", "median"]:
        dummy = DummyRegressor(strategy=baseline_type)
        dummy.fit(np.ones((len(y_train), 1)), y_train)
        y_pred_baseline = dummy.predict(np.ones((len(y_test), 1)))
        return r2_score(y_test, y_pred_baseline)

    else:
        raise ValueError(f"Unknown baseline_type: {baseline_type}")


def calculate_baseline_r2_circular(y_train, y_test, y_pred=None, baseline_type="mean", n_shuffles=None, random_state=None):
    if n_shuffles is not None :
        raise ValueError("Shuffles not supported for circular baseline R2.")
    
    y_train_cos, y_train_sin = y_train[:, 0], y_train[:, 1]
    y_test_cos, y_test_sin = y_test[:, 0], y_test[:, 1]

    dummy_cos = DummyRegressor(strategy=baseline_type)
    dummy_cos.fit(np.ones((len(y_train_cos), 1)), y_train_cos)
    y_pred_cos = dummy_cos.predict(np.ones((len(y_test_cos), 1)))

    dummy_sin = DummyRegressor(strategy=baseline_type)
    dummy_sin.fit(np.ones((len(y_train_sin), 1)), y_train_sin)
    y_pred_sin = dummy_sin.predict(np.ones((len(y_test_sin), 1)))

    return circular_r2_score(y_test_cos, y_test_sin, y_pred_cos, y_pred_sin)


def calculate_baseline_r2_boolean(
    y_train,
    y_test,
    y_pred_prob,
    n_shuffles=30,
    random_state=None,
    baseline_type="mean"
):
    rng = np.random.RandomState(random_state)

    if baseline_type == "mean":
        # Predict constant probability = positive class fraction
        p_pos = np.mean(y_train)
        y_pred_baseline = np.full_like(y_test, fill_value=p_pos, dtype=np.float32)
        try:
            return roc_auc_score(y_test, y_pred_baseline)
        except ValueError:
            return np.nan  # One class in y_test only

    elif baseline_type == "shuffle":
        scores = []
        for _ in range(n_shuffles):
            shuffled = y_pred_prob.copy()
            rng.shuffle(shuffled)
            try:
                auc = roc_auc_score(y_test, shuffled)
            except ValueError:
                auc = np.nan
            scores.append(auc)
        return np.nanmean(scores)

    else:
        raise ValueError(f"Unknown baseline_type: {baseline_type}")


def calculate_baseline_r2_count_deviance(y_train, y_test, y_pred=None, baseline_type="mean", n_shuffles=30, random_state=None):
    rng = np.random.RandomState(random_state)

    def poisson_deviance(y_true, y_pred):
        y_true = np.clip(y_true, 1e-6, None)
        y_pred = np.clip(y_pred, 1e-6, None)
        return np.sum(2 * (y_true * np.log(y_true / y_pred + 1e-8) - (y_true - y_pred)))

    y_test_mean = np.clip(np.mean(y_test), 1e-6, None)

    if baseline_type == "shuffle":
        if y_pred is None:
            raise ValueError("y_pred must be provided for shuffle baseline.")
        scores = []
        for _ in range(n_shuffles):
            shuffled = y_pred.copy()
            rng.shuffle(shuffled)
            dev_model = poisson_deviance(y_test, shuffled)
            dev_null = poisson_deviance(y_test, y_test_mean)
            r2 = 1 - dev_model / dev_null if dev_null != 0 else 0.0
            scores.append(r2)
        return np.mean(scores)

    elif baseline_type in ["mean", "median"]:
        y_pred_baseline = np.full_like(y_test, y_test_mean)
        dev_model = poisson_deviance(y_test, y_pred_baseline)
        dev_null = poisson_deviance(y_test, y_test_mean)
        return 1 - dev_model / dev_null if dev_null != 0 else 0.0

    else:
        raise ValueError(f"Unknown baseline_type: {baseline_type}")


###### REGRESSION FUNCTIONS ######
def regress_scalar(
    train_df,
    test_df,
    feature_name,
    min_samples,
    n_baseline_shuffles,
    random_state,
    clip_to_01=False,
):
    X_train, y_train, X_test, y_test = extract_valid_data(
        train_df, test_df, feature_name
    )
    if len(y_train) < min_samples:
        print(
            f"Not enough samples for regression on feature '{feature_name}': {len(y_train)} < {min_samples}"
        )
        return None, None

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_train = model.predict(X_train)
    if clip_to_01:
        y_pred = np.clip(y_pred, 0.0, 1.0)

    perf_train = model.score(X_train, y_train)
    perf_test = model.score(X_test, y_test)
    perf_baseline = calculate_baseline_r2_scalar(
        y_train=y_train, y_test=y_test, y_pred=y_pred, baseline_type="mean",
    )
    mse = mean_squared_error(y_test, y_pred)

    model_data = {
        "X_train": X_train,
        "y_train": y_train,
        "y_pred_train": y_pred_train,
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": y_pred,
        "perf_test": perf_test,
        "perf_baseline": perf_baseline,
    }

    result = {
        "feature": feature_name,
        "perf_train": perf_train,
        "perf_test": perf_test,
        "perf_test_baseline": perf_baseline,
        "perf_test_normalized": perf_test - perf_baseline,
        "N_train": len(y_train),
        "N_test": len(y_test),
        "y_sd_train": y_train.std(),
        "y_sd_test": y_test.std(),
        "MSE": mse,
    }

    return result, model_data


def regress_circular(
    train_df, test_df, feature_name, min_samples, n_baseline_shuffles, random_state
):
    train_df, test_df = extract_valid_data(
        train_df, test_df, feature_name, return_clean_df=True
    )

    cos_f, sin_f = f"{feature_name}_cos", f"{feature_name}_sin"
    # if cos_f not in train_df.columns or sin_f not in train_df.columns:
    #     print(f"[INFO] Feature '{feature_name}' with circular components not found in DataFrame. Adding now")
    train_df[cos_f] = np.cos(train_df[feature_name])
    train_df[sin_f] = np.sin(train_df[feature_name])
    test_df[cos_f] = np.cos(test_df[feature_name])
    test_df[sin_f] = np.sin(test_df[feature_name])
    train_df, test_df = extract_valid_data(
        train_df, test_df, [cos_f, sin_f], return_clean_df=True
    )

    result_cos, data_cos = regress_scalar(
        train_df, test_df, cos_f, min_samples, n_baseline_shuffles, random_state
    )
    result_sin, data_sin = regress_scalar(
        train_df, test_df, sin_f, min_samples, n_baseline_shuffles, random_state
    )

    if result_cos is None or result_sin is None:
        return None, None

    y_cos_test = data_cos["y_test"]
    y_sin_test = data_sin["y_test"]
    y_pred_cos = data_cos["y_pred"]
    y_pred_sin = data_sin["y_pred"]

    if len(y_cos_test) != len(y_sin_test) or len(y_pred_cos) != len(y_pred_sin):
        print(
            f"Lengths for feature '{feature_name}': \n"
            f"y_cos_test={len(y_cos_test)}, y_sin_test={len(y_sin_test)},\n"
            f"y_pred_cos={len(y_pred_cos)}, y_pred_sin={len(y_pred_sin)},\n"
        )

    perf_train = circular_r2_score(
        data_cos["y_train"],
        data_sin["y_train"],
        data_cos["y_pred_train"],
        data_sin["y_pred_train"],
    )
    perf_test = circular_r2_score(y_cos_test, y_sin_test, y_pred_cos, y_pred_sin)
    perf_baseline = calculate_baseline_r2_circular(
        np.column_stack((y_cos_test, y_sin_test)),
        np.column_stack((y_cos_test, y_sin_test)),
        np.column_stack((y_pred_cos, y_pred_sin)),
    )

    return {
        "feature": feature_name,
        "perf_train": perf_train,
        "perf_test": perf_test,
        "perf_test_baseline": perf_baseline,
        "perf_test_normalized": perf_test - perf_baseline,
        "N_train": result_cos["N_train"],
        "N_test": result_cos["N_test"],
        "y_sd_train": result_cos["y_sd_train"],
        "y_sd_test": result_cos["y_sd_test"],
        "MSE": (result_cos["MSE"] + result_sin["MSE"]) / 2,
    }, {
        "X_test": data_cos["X_test"],
        "y_test_cos": y_cos_test,
        "y_test_sin": y_sin_test,
        "y_pred_cos": y_pred_cos,
        "y_pred_sin": y_pred_sin,
        "perf_test": perf_test,
        "perf_baseline": perf_baseline,
    }


def regress_vector(
    train_df, test_df, feature_name, min_samples, n_baseline_shuffles, random_state
):
    """
    Regress a feature that is represented as a vector with two components.
    This function expects the feature to have two columns with suffixes '_x' and '_y'.
    """

    suffixes = ["_x", "_y"]
    # Confirm that the colums for the suffixes exist
    if not all(feature_name + suffix in train_df.columns for suffix in suffixes):
        print(f"Feature '{feature_name}' with vector suffixes not found in DataFrame.")
        return None, None

    results = []
    mse_total, perf_total, perf_baseline_total = 0, 0, 0
    N_total = 0

    model_data = {"is_vector": True, "components": {}}

    for suffix in suffixes:
        f = feature_name + suffix
        result, comp_data = regress_scalar(
            train_df, test_df, f, min_samples, n_baseline_shuffles, random_state
        )
        if result is None:
            continue
        results.append(result)
        model_data["components"][f] = comp_data
        mse_total += result["MSE"]
        perf_total += result["perf_test"]
        perf_baseline_total += result["perf_test_baseline"]
        N_total += result["N_test"]

    if not results:
        return None, None

    return {
        "feature": feature_name,
        "perf_train": np.mean([r["perf_train"] for r in results]),
        "perf_test": perf_total / len(results),
        "perf_test_baseline": perf_baseline_total / len(results),
        "perf_test_normalized": (perf_total - perf_baseline_total) / len(results),
        "N_train": sum(r["N_train"] for r in results),
        "N_test": N_total,
        "y_sd_train": np.mean([r["y_sd_train"] for r in results]),
        "y_sd_test": np.mean([r["y_sd_test"] for r in results]),
        "MSE": mse_total / len(results),
    }, model_data


def regress_boolean(
    train_df, test_df, feature_name, min_samples, n_baseline_shuffles, random_state
):
    X_train, y_train, X_test, y_test = extract_valid_data(
        train_df, test_df, feature_name
    )
    if len(y_train) < min_samples:
        return None, None

    model = LogisticRegression(max_iter=1000, random_state=random_state)
    model.fit(X_train, y_train)
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    y_pred_class = model.predict(X_test)

    try:
        perf_test = roc_auc_score(y_test, y_pred_prob)
    except ValueError:
        perf_test = np.nan  # If only one class present in y_test

    perf_baseline = calculate_baseline_r2_boolean(
        y_train, y_test, y_pred_prob, n_baseline_shuffles, random_state
    )
    logloss = log_loss(y_test, y_pred_prob)

    return {
        "feature": feature_name,
        "perf_train": model.score(X_train, y_train),
        "perf_test": perf_test,
        "perf_test_baseline": perf_baseline,
        "perf_test_normalized": perf_test - perf_baseline,
        "N_train": len(y_train),
        "N_test": len(y_test),
        "y_sd_train": y_train.std(),
        "y_sd_test": y_test.std(),
        "MSE": logloss,
    }, {
        "X_test": X_test,
        "y_test": y_test,
        "y_pred_prob": y_pred_prob,
        "y_pred_class": y_pred_class,
        "perf_test": perf_test,
        "perf_baseline": perf_baseline,
        "is_boolean": True,
    }


def regress_count(train_df, 
                  test_df, 
                  feature_name, 
                  min_samples=10, 
                  n_baseline_shuffles=0, 
                  random_state=42, 
                  perf_type=["deviance", "variance"][-1],
                  verbose=0,
                  ):
    if n_baseline_shuffles != 0:
        print(f"[ERROR] n_baseline_shuffles is not implemented")
        return None, None

    X_train, y_train, X_test, y_test = extract_valid_data(train_df, test_df, feature_name)

    if verbose > 0:
        print(f"[DEBUG] X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")

    if len(y_train) < min_samples:
        print("[ERROR] Not enough training samples")
        return None, None

    if (y_train < 0).any() or not np.all(np.equal(np.mod(y_train, 1), 0)):
        print(f"[WARNING] Invalid count values in '{feature_name}' — skipping.")
        return None, None

    try:
        mean_y = np.mean(y_train)
        var_y = np.var(y_train)
        vmr = var_y / (mean_y + 1e-6)
        is_overdispersed = vmr > 1.5

        family = sm.families.NegativeBinomial() if is_overdispersed else sm.families.Poisson()
        family_name = "Negative Binomial" if is_overdispersed else "Poisson"
        print(f"[INFO] Using {family_name} regression for '{feature_name}' (VMR = {vmr:.2f})")

        X_train_sm = sm.add_constant(X_train, has_constant="add")
        X_test_sm = sm.add_constant(X_test, has_constant="add")

        model = sm.GLM(y_train, X_train_sm, family=family)
        results = model.fit(method='newton', maxiter=100, disp=False)

        y_pred = np.clip(results.predict(X_test_sm), 1e-6, None)
        y_mean = np.clip(np.mean(y_test), 1e-6, None)

        deviance_model = np.sum(2 * (y_test * np.log(y_test / y_pred + 1e-8) - (y_test - y_pred)))
        deviance_null = np.sum(2 * (y_test * np.log(y_test / y_mean + 1e-8) - (y_test - y_mean)))
        perf_test_deviance = 1 - deviance_model / deviance_null if deviance_null != 0 else 0.0
        perf_test_variance = r2_score(y_test, y_pred)

        if verbose > 0:
            print(f"[DEBUG] perf_test (deviance R²): {perf_test_deviance:.4f}")
            print(f"[DEBUG] perf_test (variance R²): {perf_test_variance:.4f}")

        y_train_pred = np.clip(results.predict(X_train_sm), 1e-6, None)
        y_train_mean = np.clip(np.mean(y_train), 1e-6, None)
        perf_train_r2_variance = r2_score(y_train, y_train_pred)

        deviance_model_train = np.sum(
            2 * (y_train * np.log(y_train / y_train_pred + 1e-8) - (y_train - y_train_pred))
        )
        deviance_null_train = np.sum(
            2 * (y_train * np.log(y_train / y_train_mean + 1e-8) - (y_train - y_train_mean))
        )
        perf_train_r2_deviance = 1 - deviance_model_train / deviance_null_train if deviance_null_train != 0 else 0.0

        if verbose > 0:
            print(f"[DEBUG] perf_train (deviance R²): {perf_train_r2_deviance:.4f}")
            print(f"[DEBUG] perf_train (variance R²): {perf_train_r2_variance:.4f}")

        # Compute baseline R² (for variance)
        perf_test_variance_baseline = calculate_baseline_r2_scalar(
            y_train=y_train,
            y_test=y_test,
            y_pred=y_pred,
            baseline_type="mean",
        )

        # Optionally: compute baseline R² (for deviance)
        perf_test_deviance_baseline = calculate_baseline_r2_count_deviance(
            y_train=y_train,
            y_test=y_test,
            y_pred=y_pred,
            baseline_type="mean",
        )

        # Normalized R² by subtracting baseline
        perf_test_normalized = (
            perf_test_variance - perf_test_variance_baseline
            if perf_type == "variance"
            else perf_test_deviance - perf_test_deviance_baseline
        )


        return {
            "feature": feature_name,
            "model_family": family_name,
            "perf_train": perf_train_r2_deviance if perf_type == "deviance" else perf_train_r2_variance,
            "perf_test": perf_test_deviance if perf_type == "deviance" else perf_test_variance,
            "perf_test_normalized": perf_test_normalized,
            "perf_test_baseline": perf_test_variance_baseline if perf_type == "variance" else perf_test_deviance_baseline,
            "N_train": len(y_train),
            "N_test": len(y_test),
            "y_sd_train": y_train.std(),
            "y_sd_test": y_test.std(),
            "MSE": mean_squared_error(y_test, y_pred),
        }, {
            "X_test": X_test,
            "y_test": y_test,
            "y_pred": y_pred,
            "perf_test": perf_test_deviance if perf_type == "deviance" else perf_test_variance,
            "perf_baseline": perf_test_variance_baseline if perf_type == "variance" else perf_test_deviance_baseline,
            "is_count": True,
            "model_family": family_name,
        }

    except Exception as e:
        print(f"[ERROR] Count regression failed for {feature_name}: {e}")
        return None, None


def run_regression_for_features(
    train_df,
    test_df,
    features,
    metadata,
    min_samples=10,
    n_baseline_shuffles=30,
    random_state=42,
):
    results = []
    models_data = {}

    for feat in features:
        try:
            config = metadata.get(feat, {})
            ftype = config.get("feature_type", "scalar")

            if ftype == "scalar":
                result, model_data = regress_scalar(
                    train_df,
                    test_df,
                    feat,
                    min_samples,
                    n_baseline_shuffles,
                    random_state,
                )
            elif ftype == "vector":
                result, model_data = regress_vector(
                    train_df,
                    test_df,
                    feat,
                    min_samples,
                    n_baseline_shuffles,
                    random_state,
                )
            elif ftype == "circular":
                result, model_data = regress_circular(
                    train_df,
                    test_df,
                    feat,
                    min_samples,
                    n_baseline_shuffles,
                    random_state,
                )
            elif ftype == "probability":
                result, model_data = regress_scalar(
                    train_df,
                    test_df,
                    feat,
                    min_samples,
                    n_baseline_shuffles,
                    random_state,
                    clip_to_01=True,
                )
            elif ftype == "boolean":
                result, model_data = regress_boolean(
                    train_df,
                    test_df,
                    feat,
                    min_samples,
                    n_baseline_shuffles,
                    random_state,
                )
            elif ftype == "count":
                result, model_data = regress_count(
                    train_df,
                    test_df,
                    feat,
                    min_samples,
                    n_baseline_shuffles,
                    random_state,
                )
            else:
                print(f"[WARN] Unknown feature_type '{ftype}' for {feat}. Skipping.")
                continue

            if result:
                result["feature"] = feat
                result["ftype"] = ftype
                results.append(result)
                models_data[feat] = model_data

            print(
                f"    [INFO] Processed {feat} with type '{ftype}', perf_test_normalized: {result['perf_test_normalized']:.3f} (N_train={result['N_train']}, N_test={result['N_test']})"
            )
        except Exception as e:
            print(f"    [WARN] Skipping {feat} with type '{ftype}' due to error: {e}")
            # traceback.print_exc()

    return (
        pd.DataFrame(results).sort_values("perf_test_normalized", ascending=False),
        models_data,
    )


def run_modular_regression_pipeline(
    dff_filtered,
    regression_features,
    feature_metadata,
    sensor_ranges,
    group_cols=["env_id", "episode_index"],
    n_splits=5,
    random_state=42,
    min_samples=1000,
    n_baseline_shuffles=0,  # using mean‐based baseline
):
    """
    K‐fold modular regression over groups (env_id, episode_index).
    Uses GroupKFold to ensure no group is split between train and test.
    Returns per‐range DataFrames of mean±std metrics for error bars.
    """
    reports = {}
    all_models_and_data = {}

    # set up GroupKFold
    

    for (min_dist, max_dist), range_name, range_abbrev, dist_col in sensor_ranges:
        print(f"\n[INFO] Processing range: {range_name} ({min_dist},{max_dist}) on '{dist_col}'")
        # filter to this range
        df_range = dff_filtered[
            (dff_filtered[dist_col] >= min_dist) &
            (dff_filtered[dist_col] <  max_dist)
        ]
        if df_range.empty:
            print(f"  [WARN] no samples for {range_name}")
            continue

        # build group labels so entire (env,episode) stay together
        group_labels = df_range[group_cols].apply(tuple, axis=1)
        if group_labels.nunique() < n_splits:
            print(f"  [WARN] Not enough unique groups for {range_name} ({group_labels.nunique()} < {n_splits})")
            print(f"  [INFO] Using {group_labels.nunique()} groups instead of {n_splits}")
            gkf = GroupKFold(n_splits=group_labels.nunique())
        else:
            gkf = GroupKFold(n_splits=n_splits)

        fold_results = []
        # perform K-fold splits
        for fold_idx, (train_idx, test_idx) in enumerate(
            gkf.split(df_range, groups=group_labels)
        ):
            print(f"  [INFO] Processing fold {fold_idx + 1}/{n_splits} for {range_name}")
            try:
                train_df = df_range.iloc[train_idx]
                test_df  = df_range.iloc[test_idx]

                # run your per‐feature regression on this fold
                fold_res, _ = run_regression_for_features(
                    train_df,
                    test_df,
                    features=regression_features,
                    metadata=feature_metadata,
                    min_samples=min_samples,
                    n_baseline_shuffles=n_baseline_shuffles,
                    random_state=random_state,
                )

                if fold_res is None or fold_res.empty:
                    print(f"  [WARN] fold {fold_idx} empty for {range_name}")
                    continue

                fold_res = fold_res.copy()
                fold_res["fold"] = fold_idx
                fold_results.append(fold_res)

            except Exception as e:
                print(f"  [ERROR] fold {fold_idx} failed for {range_name}: {e}")
                continue

        if not fold_results:
            print(f"  [WARN] no successful folds for {range_name}")
            continue

        # aggregate metrics across folds (mean ± std)
        all_folds = pd.concat(fold_results, ignore_index=True)

        agg = all_folds.groupby("feature").agg(
            perf_test_mean    = ("perf_test", "mean"),
            perf_test_std     = ("perf_test", "std"),
            baseline_mean     = ("perf_test_baseline", "mean"),
            baseline_std      = ("perf_test_baseline", "std"),
            norm_test_mean    = ("perf_test_normalized", "mean"),
            norm_test_std     = ("perf_test_normalized", "std"),
            MSE_mean          = ("MSE", "mean"),
            MSE_std           = ("MSE", "std"),
            N_test_mean       = ("N_test", "mean"),
        ).reset_index()

        agg["ftype"] = agg["feature"].map(
            lambda f: feature_metadata.get(f, {}).get("feature_type", "unknown")
        )

        key = f"{dist_col.split('_')[-1]}_{range_abbrev}"
        reports[key] = agg

    return reports, all_models_and_data


def plot_perf_test_normalized(
    reports,
    feature_metadata,
    outfile_base=None,
    perf_threshold=0.0,
    exclusion_list=None,
    name_key="name",
    color_map=None,
    sort_by_ftype=True,
    ftype_order=None,
):
    if color_map is None:
        color_map = {"unknown": "#7f7f7f"}

    for key, df in reports.items():
        # figure out which “normalized‐performance” columns we have
        if "perf_test_normalized" in df.columns:
            mean_col = "perf_test_normalized"
            std_col = None
        elif "norm_test_mean" in df.columns:
            mean_col = "norm_test_mean"
            std_col  = "norm_test_std"
        else:
            raise ValueError("Don’t recognize normalized‐perf columns in reports")

        # apply threshold
        df_plot = df[df[mean_col] > perf_threshold].copy()
        if df_plot.empty:
            continue

        # optional exclusion
        if exclusion_list:
            df_plot = df_plot[~df_plot["feature"].isin(exclusion_list)]
            if df_plot.empty:
                continue

        # sort
        if sort_by_ftype and ftype_order:
            rank = {f:i for i,f in enumerate(ftype_order)}
            df_plot["ftype_rank"] = df_plot["ftype"].map(rank).fillna(len(rank))
            df_plot = df_plot.sort_values(["ftype_rank", mean_col],
                                          ascending=[True, False])
        else:
            df_plot = df_plot.sort_values(mean_col, ascending=False)

        # names & colors
        labels = [feature_metadata.get(f,{}).get(name_key,f)
                  for f in df_plot["feature"]]
        ftypes = df_plot["ftype"].fillna("unknown").tolist()
        colors = [color_map.get(ft, color_map["unknown"]) for ft in ftypes]

        # plot
        y = np.arange(len(df_plot))
        plt.figure(figsize=(6, max(4, len(df_plot)*0.25)))
        if std_col:
            # new mode: show error bars
            plt.barh(y, df_plot[mean_col], xerr=df_plot[std_col], color=colors, alpha=0.8)
        else:
            # old mode: plain bars
            plt.barh(y, df_plot[mean_col], color=colors)

        plt.yticks(y, labels)
        plt.xlabel("Normalized Test Performance")
        plt.title(key)
        plt.xlim(0, 1)
        plt.gca().invert_yaxis()

        # legend
        uniq = list(dict.fromkeys(ftypes))
        handles = [mpatches.Patch(color=color_map.get(ft,"#777"), label=ft)
                   for ft in uniq if ft in ftype_order] if ftype_order else \
                  [mpatches.Patch(color=color_map.get(ft,"#777"), label=ft) for ft in uniq]
        plt.legend(handles=handles, title="Feature Type", fontsize="small", loc="lower right")

        plt.tight_layout()
        if outfile_base:
            plt.savefig(f"{outfile_base}_{key}_perf_test_normalized.png", bbox_inches="tight")
            print(f"Saved plot to {outfile_base}_{key}_perf_test_normalized.png")
        else:
            plt.show()
        plt.close()


def save_regression_report_modular(
    all_reg_reports, outfile_base
):
    for range_name, df in all_reg_reports.items():
        # df = df.round(3)
        fname = f"{outfile_base}_regression_modular_results_{range_name}.csv"
        df.to_csv(fname, index=False, float_format="%.3f")
        print(f"Saved regression report for {range_name} to {fname}")

    plot_perf_test_normalized(all_reg_reports,
                                feature_metadata=FEATURE_METADATA,
                                outfile_base=outfile_base,
                                perf_threshold=0.0,
                                exclusion_list=None, # TODO
                                name_key="name",
                                color_map=FEATURE_TYPE_COLORMAP,
                                sort_by_ftype=True,
                                ftype_order=["scalar", "vector", "circular", "probability", "boolean", "count", "other"],
                                )
    print(f"Saved all regression plots under {outfile_base}")



####### GROUPED DECODING FUNCTIONS #######
def get_per_agent_regression_mode(metadata):
    """
    Determine regression mode based on `agent_size_mode` and number of active agents.
    Returns one of: "agent_id", "pairwise_bins", or "bins".
    """
    agent_size_mode = metadata["all_args"]["agent_size_mode"]
    active_ids = metadata.get("multi_agent_args", {}).get("active_agent_ids", [])
    # regex match A[something]B (e.g. AltB, A5ltB, AeqB, etc.)
    pattern = re.compile(r"^A.*B$")
    if "fixed" in agent_size_mode or pattern.match(agent_size_mode):
        return "agent_id"
    else:
        if len(active_ids) == 2:
            return "pairwise_bins"
        return "bins"


def get_group_info(df, metadata, bins=4, pairwise_bins=2):
    """
    Determine grouping mode and split DataFrame into groups.
    Returns: groups dict, group_sizes dict, mode string.
    """
    df = df.copy()

    active_ids = metadata.get("multi_agent_args", {}).get("active_agent_ids", [])
    mode = get_per_agent_regression_mode(metadata)
    groups, group_sizes = {}, {}

    if mode == "agent_id":
        for agent in active_ids:
            key = f"agent{agent}"
            sub = df[df["agent_id"] == agent].copy()
            groups[key] = sub
            group_sizes[key] = len(sub)

    elif mode == "bins":
        edges = np.linspace(0.0, 1.0, bins + 1)
        labels = [f"{edges[i]:.2f}-{edges[i+1]:.2f}" for i in range(bins)]
        df["size_bin"] = pd.cut(df["agent_size"], bins=edges, labels=labels,
                                  include_lowest=True, right=False)
        for lbl in labels:
            sub = df[df["size_bin"] == lbl]
            if not sub.empty:
                groups[lbl] = sub.copy()
                group_sizes[lbl] = len(sub)

    elif mode == "pairwise_bins":
        if len(active_ids) != 2:
            raise ValueError("pairwise_bins requires exactly two agents")
        dfs = df[df["agent_id"].isin(active_ids)]
        left = dfs[["episode_index","env_id","time_step","agent_id","agent_size"]]
        right = left.rename(columns={"agent_id":"other_agent_id","agent_size":"other_agent_size"})
        merged = pd.merge(left, right, on=["episode_index","env_id","time_step"])
        merged = merged[merged["agent_id"] != merged["other_agent_id"]]
        edges = np.linspace(0.0, 1.0, pairwise_bins + 1)
        lbls = [f"{edges[i]:.2f}-{edges[i+1]:.2f}" for i in range(pairwise_bins)]
        for side, col in [("self", "agent_size"), ("other", "other_agent_size")]:
            merged[f"{side}_bin"] = pd.cut(merged[col], bins=edges, labels=lbls,
                                            include_lowest=True, right=False)
        merged = merged.dropna(subset=["self_bin","other_bin"])
        merged["pair_bin"] = merged["self_bin"].astype(str) + "__" + merged["other_bin"].astype(str)
        for pair in merged["pair_bin"].unique():
            sub = merged[merged["pair_bin"] == pair]
            if not sub.empty:
                groups[pair] = sub.copy()
                group_sizes[pair] = len(sub)

    else:
        raise ValueError(f"Unknown mode: {mode}")

    return groups, group_sizes, mode


def plot_grouped_agent_decoding(
    results_by_group, outfile_base, group_mode, group_sizes, agent_id_to_size=None
):
    all_range_keys = list(next(iter(results_by_group.values())).keys())

    for range_key in all_range_keys:
        df_list = []
        for group_key, range_dict in results_by_group.items():
            if range_key not in range_dict:
                continue
            df = range_dict[range_key].copy()

            # Label construction
            if group_mode == "agent_id" and agent_id_to_size:
                label = f"{group_key} (size {agent_id_to_size.get(int(group_key.replace('agent','')), '?')})"
            else:
                label = group_key

            df["group_label"] = label
            df_list.append(df)

        combined_df = pd.concat(df_list, ignore_index=True)

        # ——— choose whichever normalized‐R² column you have ———
        if "perf_test_normalized" in combined_df.columns:
            val_col = "perf_test_normalized"
        elif "norm_test_mean" in combined_df.columns:
            val_col = "norm_test_mean"
        else:
            raise KeyError(
                "Neither 'perf_test_normalized' nor 'norm_test_mean' found in combined_df"
            )

        # Pivot so rows=feature, columns=group_label
        pivot_df = combined_df.pivot_table(
            index="feature",
            columns="group_label",
            values=val_col,
            aggfunc="mean",
        ).fillna(0)

        # Sort by average across groups
        pivot_df["mean"] = pivot_df.mean(axis=1)
        pivot_df = pivot_df.sort_values("mean", ascending=True).drop(columns=["mean"])

        # Dynamic figure height
        num_features = len(pivot_df)
        fig_height = max(0.35 * num_features, 6)
        fig, ax = plt.subplots(
            figsize=(max(12, len(pivot_df.columns) * 1.8), fig_height)
        )

        pivot_df.plot(kind="barh", ax=ax, width=0.8)

        ax.set_xlabel("Performance Normalized")
        ax.set_ylabel("Feature")
        ax.set_yticklabels(pivot_df.index, fontsize=10)
        ax.invert_yaxis()
        ax.set_xlim(0, 1.0)


        # Build subtitle with sample size
        n_info = []
        for label in pivot_df.columns:
            if group_mode == "agent_id":
                raw_key = label.split(" ")[0]
            else:
                raw_key = label
            n_info.append(f"{raw_key} n={group_sizes.get(raw_key, 0)}")
        subtitle = " | ".join(n_info)

        ax.set_title(
            f"R² Test Normalized by Feature and Group\n{range_key}\n{subtitle}"
        )
        ax.legend(title="Group", bbox_to_anchor=(1.05, 1), loc="upper left")

        # Improve spacing
        plt.subplots_adjust(left=0.25, right=0.85, top=0.92, bottom=0.05)

        save_path = (
            f"{outfile_base}_{range_key}_grouped_{group_mode}_bar_horizontal.png"
        )
        plt.savefig(save_path, dpi=300)
        plt.close()


# -----------------------------------------------------------------------------
# Helper utilities
# -----------------------------------------------------------------------------

def _adjust_color(hex_color: str, factor: float) -> str:
    """Return ``hex_color`` darkened (*factor*<1) or lightened (*factor*>1).

    Parameters
    ----------
    hex_color : str
        Base colour in any matplotlib‑parsable format (e.g. "#1f77b4").
    factor : float
        1.0 → unchanged.  0.0 → black.  >1 → towards white, <1 → darker.
    """
    rgb = np.array(mcolors.to_rgb(hex_color))
    if factor < 1:
        rgb = rgb * factor  # darken by scaling
    else:
        rgb = rgb + (1 - rgb) * (factor - 1)  # lighten by blending towards white
    rgb = np.clip(rgb, 0, 1)
    return mcolors.to_hex(rgb)


# -----------------------------------------------------------------------------
# Main plotting routine
# -----------------------------------------------------------------------------

def plot_grouped_agent_decoding(
    results_by_group: Dict[str, Dict[str, pd.DataFrame]],
    outfile_base: str,
    group_mode: str,
    group_sizes: Dict[str, int],
    feature_metadata: Dict[str, Dict[str, str]],
    perf_threshold: float = 0.0,
    exclusion_list: Optional[List[str]] = None,
    name_key: str = "name",
    color_map: Optional[Dict[str, str]] = None,
    ftype_order: Optional[List[str]] = None,
    agent_id_to_size: Optional[Dict[int, int]] = None,
):
    """Grouped *normalized* performance plot with **group‑specific shading**.

    Each *feature* is plotted on its own row, with *k* adjacent bars (one per
    ``group_key``).  Base hue derives from ``color_map`` by **feature‑type**; the
    *shade* (light ↔ dark) encodes the **group identity** so that, for example, the
    first group is drawn in lighter tones and progressively darker shades are used
    for subsequent groups.  Error bars are included automatically when
    ``*_std`` columns are present.

    The function otherwise mirrors the behaviour of ``plot_perf_test_normalized``:
    ordering by ``ftype_order`` (if supplied) and intra‑type sorting by *maximum*
    performance across groups.
    """

    # ------------------------------------------------------------------
    # 0.  Configuration & probe
    # ------------------------------------------------------------------
    if color_map is None:
        color_map = {"unknown": "#7f7f7f"}

    probe_df = next(iter(next(iter(results_by_group.values())).values()))
    if "perf_test_normalized" in probe_df.columns:
        mean_col, std_col = "perf_test_normalized", None
    elif "norm_test_mean" in probe_df.columns:
        mean_col, std_col = "norm_test_mean", "norm_test_std"
    else:
        raise KeyError("No recognised normalised‑performance columns found.")

    all_range_keys = list(next(iter(results_by_group.values())).keys())

    # Feature‑type rank mapping
    if ftype_order:
        ftype_rank_map = {ft: i for i, ft in enumerate(ftype_order)}
    else:
        ftype_rank_map = {}

    # Pre‑compute group‑specific shade factors (light → dark)
    n_groups_total = len(next(iter(results_by_group.values())))
    # Ensure reproducible ordering of groups for colour assignment
    all_group_labels_sorted = sorted({
        (g if group_mode != "agent_id" else f"{g}") for g in results_by_group.keys()
    })
    shade_factors = np.linspace(1.0, 0.6, len(all_group_labels_sorted))  # 1 → original, 0.6 → darker
    shade_by_group = {label: factor for label, factor in zip(all_group_labels_sorted, shade_factors)}

    # ------------------------------------------------------------------
    # Iterate over range windows
    # ------------------------------------------------------------------
    for range_key in all_range_keys:
        long_frames: List[pd.DataFrame] = []

        # ------------------------- gather ------------------------------
        for group_key, range_dict in results_by_group.items():
            if range_key not in range_dict:
                continue
            df = range_dict[range_key].copy()

            # Filtering
            df = df[df[mean_col] > perf_threshold]
            if exclusion_list is not None:
                df = df[~df["feature"].isin(exclusion_list)]
            if df.empty:
                continue

            # Attach ftype if missing
            if "ftype" not in df.columns:
                df["ftype"] = df["feature"].map(
                    lambda f: feature_metadata.get(f, {}).get("ftype", "unknown")
                )

            # Human‑friendly group label
            if group_mode == "agent_id" and agent_id_to_size:
                raw_id = int(group_key.replace("agent", ""))
                label = f"{group_key} (size {agent_id_to_size.get(raw_id, '?')})"
            else:
                label = group_key
            df["group_label"] = label

            long_frames.append(df)

        if not long_frames:
            continue

        combined = pd.concat(long_frames, ignore_index=True)

        # Build ftype rank on the fly if necessary
        if not ftype_rank_map:
            for ft in combined["ftype"].dropna().unique():
                if ft not in ftype_rank_map:
                    ftype_rank_map[ft] = len(ftype_rank_map)

        # Intra‑type ordering by max perf
        feat_max = combined.groupby("feature")[mean_col].max()
        combined["ftype_rank"] = combined["ftype"].map(ftype_rank_map)
        combined["feature_max"] = combined["feature"].map(feat_max)

        order_df = combined[["feature", "ftype_rank", "feature_max"]].drop_duplicates()
        order_df = order_df.sort_values(["ftype_rank", "feature_max"], ascending=[True, False])
        feature_order = order_df["feature"].tolist()

        # ------------------- pivot to wide -----------------------------
        pivot_mean = (
            combined.pivot_table(index="feature", columns="group_label", values=mean_col, aggfunc="mean")
            .reindex(feature_order)
            .fillna(0)
        )
        if std_col:
            pivot_std = (
                combined.pivot_table(index="feature", columns="group_label", values=std_col, aggfunc="mean")
                .reindex(feature_order)
                .fillna(0)
            )[pivot_mean.columns]
        else:
            pivot_std = None

        # Map feature → ftype for colour lookup
        feature_ftype = combined.drop_duplicates("feature").set_index("feature")["ftype"].to_dict()

        # ------------------- manual grouped barh -----------------------
        n_feat = len(pivot_mean)
        n_groups = len(pivot_mean.columns)
        base_ys = np.arange(n_feat)
        total_height = 0.8
        sub_h = total_height / n_groups
        offsets = np.linspace(-total_height / 2 + sub_h / 2, total_height / 2 - sub_h / 2, n_groups)

        fig_h = max(0.35 * n_feat, 6)
        fig_w = max(12, n_groups * 1.8)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))

        for gi, (group_label, offset) in enumerate(zip(pivot_mean.columns, offsets)):
            factor = shade_by_group.get(group_label.split(" ")[0] if group_mode == "agent_id" else group_label, 1.0)
            widths = pivot_mean[group_label].values
            if pivot_std is not None:
                errs = pivot_std[group_label].values
            else:
                errs = None

            colors = [
                _adjust_color(color_map.get(feature_ftype.get(feat, "unknown"), "#7f7f7f"), factor)
                for feat in pivot_mean.index
            ]
            y_pos = base_ys + offset
            ax.barh(y_pos, widths, height=sub_h, xerr=errs, color=colors, edgecolor="black", linewidth=0.3, label=group_label)

        # Y‑labels as nice names
        ax.set_yticks(base_ys)
        ax.set_yticklabels([feature_metadata.get(f, {}).get(name_key, f) for f in pivot_mean.index], fontsize=10)

        ax.set_xlabel("Normalized Test Performance")
        ax.set_ylabel("Feature")
        ax.set_xlim(0, 1)
        ax.invert_yaxis()

        # Subtitle with sample sizes
        subtitle_parts = []
        for label in pivot_mean.columns:
            raw_key = label.split(" ")[0] if group_mode == "agent_id" else label
            subtitle_parts.append(f"{raw_key} n={group_sizes.get(raw_key, 0)}")
        subtitle = " | ".join(subtitle_parts)

        ax.set_title(f"R² Test Normalized by Feature and Group\n{range_key}\n{subtitle}", pad=20)
        ax.legend(title="Group", bbox_to_anchor=(1.05, 1), loc="upper left")

        plt.subplots_adjust(left=0.3, right=0.85, top=0.9, bottom=0.05)

        save_path = f"{outfile_base}_{range_key}_grouped_{group_mode}_bar_horizontal.png"
        plt.savefig(save_path, dpi=300)
        plt.close()

        print(f"Saved plot to {save_path}")


def run_decoding_by_group_modular(
    df,
    outfile_base,
    metadata,
    regression_feats,
    behavior_mode="foraging",
    random_state=42,
    min_samples=10,
    n_baseline_shuffles=0,
    bins=4,
    pairwise_bins=2,
):
    """
    Group data, run modular regression per group, then summarize across groups.
    """
    base_path = Path(outfile_base)
    (base_path.parent / "by_group").mkdir(parents=True, exist_ok=True)

    # Determine groups
    groups, group_sizes, mode = get_group_info(df, metadata, bins=bins, pairwise_bins=pairwise_bins)

    # Shared pipeline params
    sensor_ranges = get_sensor_ranges(metadata, behavior_mode)

    results = {}
    merge_keys = ["episode_index", "env_id", "time_step", "agent_id"]
    for key, subset in groups.items():
        print(f"\n[INFO] Processing group '{key}' with {len(subset)} samples in mode '{mode}'")
        # No need to call feature processing again, as it's been done once already.
        df_subset = df.merge(
            subset[merge_keys],
            on=merge_keys,
            how="inner",
        )
        reports, _ = run_modular_regression_pipeline(
            df_subset,
            regression_features=regression_feats,
            feature_metadata=FEATURE_METADATA,
            sensor_ranges=sensor_ranges,
            group_cols=["env_id","episode_index"],
            random_state=random_state,
            min_samples=min_samples,
            n_baseline_shuffles=n_baseline_shuffles,
        )

        # Save per-group outputs
        group_file = base_path.parent / "by_group" / f"{base_path.name}_{key}"
        save_regression_report_modular(reports, str(group_file))

        results[key] = reports

    # Summarize across groups
    plot_grouped_agent_decoding(
        results, outfile_base, group_mode=mode, group_sizes=group_sizes
    )
    return results


def run_decoding_report_modular(
    df,
    outfile_base,
    behavior_mode="foraging",
    downsample_nrows=50000,  # None means no downsampling
):
    assert behavior_mode in (
        "homing",
        "foraging",
    ), f"Invalid behavior mode: {behavior_mode}"

    df["arena_size"] = df.groupby(["env_id", "episode_index"])["arena_size"].ffill()
    if behavior_mode == "homing":
        df = df[df["agent_id"] == 1]

    metadata = df["metadata"].iloc[0]
    active_ids = metadata.get("multi_agent_args", {}).get("active_agent_ids", [])
    df = df[df["agent_id"].isin(active_ids)]

    # Should this be done here, or for every data split? (e.g. mormyromast)
    df = stratified_downsample(
        df, downsample_nrows=downsample_nrows, random_state=42
    )

    df, feature_names = prepare_features_for_decoding(
        df, 
        normalize=False, # Shouldn't be 
        behavior_mode=behavior_mode,
    )

    # regression_features = get_regression_features_for_mode(behavior_mode, df)
    # Use all features for now (also, anyway only plotting the ones that are predicted well)
    regression_features = get_regression_features(df, feature_names)
    print(f"Using {len(regression_features)} regression features for {behavior_mode} mode.")
    print(f"Regression features: {regression_features}")

    plot_regression_feature_distributions(
        df, regression_features, output_folder=outfile_base
    )

    sensor_ranges = get_sensor_ranges(metadata, behavior_mode)

    reg_reports, all_models_and_data = run_modular_regression_pipeline(
        df,
        regression_features,
        feature_metadata=FEATURE_METADATA,
        sensor_ranges=sensor_ranges,
     )
    print(f"Generated {len(reg_reports)} regression reports.")
    save_regression_report_modular(reg_reports, outfile_base)

    print(f"Running decoding by group...")
    run_decoding_by_group_modular(df, outfile_base, metadata, regression_features, behavior_mode=behavior_mode)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run decoding report with flat pkl data file")
    parser.add_argument("--flat_pkl_file", type=str, default=None, help="Path to flattened pickle file")
    args = parser.parse_args()
    flat_pkl_file = args.flat_pkl_file

    if os.getenv("USER") == "sonja":
        # Load data
        behavior_mode = "foraging"
        if behavior_mode == "foraging":
            df = pd.read_pickle(
                "/home/sonja/marl_fish/onpolicy/custom/fish/results/rmappo-MultiAgentFishEnv-medium/20250302_202903/outputs/MAFish_neural_20250302_202903_EP78y4k4_agg_flattened.pkl"
            )
            output_folder = "/home/sonja/marl_fish/onpolicy/custom/fish/results/rmappo-MultiAgentFishEnv-medium/20250302_202903/outputs/"
            outfile_base = "/home/sonja/marl_fish/onpolicy/custom/fish/results/rmappo-MultiAgentFishEnv-medium/20250302_202903/outputs/mini_reports/MAFish_20250302_202903_EP78y4k4/MAFish_20250302_202903_EP78y4k4_"
        elif behavior_mode == "homing":
            df = pd.read_pickle(
                "/home/sonja/marl_fish/onpolicy/custom/fish/results/rmappo-MultiAgentFishEnv-homing/20250305_112012/outputs/MAFish_neural_20250305_112012_gHkAfvTy_agg_flattened.pkl"
            )
            outfile_base = "dummy_homing_dir"

        run_decoding_report_modular(df, outfile_base, behavior_mode=behavior_mode)

    if flat_pkl_file is not None:
        most_recent_file = flat_pkl_file
    elif os.getenv("USER") == "satsingh":
        DATA_DIR = "/srv/marl/satsingh/marl_fish/"
        min_mb = 5
        min_size_bytes = min_mb * 1024 * 1024  # MB
        print(
            f"Finding most recent flat-pkl file greater than {min_mb} MB in size ending in _flattened.pkl in all subfolders of {DATA_DIR}"
        )
        most_recent_file = None
        most_recent_mtime = -1
        for root, dirs, files in os.walk(DATA_DIR):
            for fname in files:
                if fname.endswith("_flattened.pkl"):
                    fpath = os.path.join(root, fname)
                    try:
                        size = os.path.getsize(fpath)
                        mtime = os.path.getmtime(fpath)
                        if size > min_size_bytes and mtime > most_recent_mtime:
                            most_recent_file = fpath
                            most_recent_mtime = mtime
                    except Exception as e:
                        print(f"Skipping {fpath}: {e}")

        if most_recent_file:
            print(f"Loading recent file >50MB: {most_recent_file} ...")
    else:
        print("No matching file found.")
        most_recent_file = None

    if most_recent_file is not None:
        start_time = time.time()
        dff = pd.read_pickle(most_recent_file)
        end_time = time.time()
        print(f"Loaded in {end_time - start_time:.2f} seconds.")
        print(f"Columns in DataFrame: {dff.columns.tolist()}")

        output_folder = "test_decoding_output/"
        os.makedirs(output_folder, exist_ok=True)

        outfile_base = f"{output_folder}/test"
        run_decoding_report_modular(
            dff,
            outfile_base,
            behavior_mode="foraging",
        )

# import os
# import glob
# import argparse
# import pandas as pd

# from cfg import FEATURE_METADATA, FEATURE_TYPE_COLORMAP

# def load_csvs(csv_dir, prefix):
#     results = {}
#     pattern = os.path.join(csv_dir, f"{prefix}_*_regression_modular_results_*.csv")
#     for fn in glob.glob(pattern):
#         base = os.path.basename(fn)
#         # split off: "<prefix>_<group>_regression_modular_results_<range>.csv"
#         after_pref = base[len(prefix) + 1:]
#         group_key, range_part = after_pref.split("_regression_modular_results_")
#         range_key = range_part.replace(".csv","")
#         df = pd.read_csv(fn)
#         results.setdefault(group_key, {})[range_key] = df
#     return results

# def infer_group_mode(groups):
#     keys = list(groups.keys())
#     if all(k.startswith("agent") for k in keys):
#         return "agent_id"
#     if any("__" in k for k in keys):
#         return "pairwise_bins"
#     return "bins"

# def compute_group_sizes(results):
#     sizes = {}
#     for g, d in results.items():
#         # take the first range’s N_test_mean as a proxy for sample size
#         df0 = next(iter(d.values()))
#         if "N_test_mean" in df0.columns:
#             sizes[g] = int(df0["N_test_mean"].mean())
#         else:
#             sizes[g] = 0
#     return sizes

# def main():
#     p = argparse.ArgumentParser(
#         description="Quickly plot grouped decoding from precomputed CSVs"
#     )
#     p.add_argument("--csv-dir",    required=True,
#                    help="Folder containing *_regression_modular_results_*.csv")
#     p.add_argument("--prefix",     required=True,
#                    help="Common prefix before _<group>_regression_modular_results_*.csv")
#     p.add_argument("--outfile-base", required=True,
#                    help="Where to save the combined grouped plot (no extension)")
#     args = p.parse_args()

#     results = load_csvs(args.csv_dir, args.prefix)
#     if not results:
#         print("⚠️  No CSVs found with that prefix!")
#         return
#     outfile_base = args.outfile_base

#     group_mode  = infer_group_mode(results)
#     group_sizes = compute_group_sizes(results)
#     plot_grouped_agent_decoding(
#         results_by_group   = results,
#         feature_metadata   = FEATURE_METADATA,
#         outfile_base       = args.outfile_base,
#         group_mode         = group_mode,
#         group_sizes        = group_sizes,
#         agent_id_to_size   = None,          # only needed for agent_id→size labels
#         perf_threshold     = 0.0,
#         name_key           = "name",
#         color_map          = FEATURE_TYPE_COLORMAP,
#         ftype_order        = ["scalar","vector","circular","probability","boolean","count","other"],
#     )
#     # plot_grouped_agent_decoding(
#     #     results, outfile_base, group_mode=group_mode, group_sizes=group_sizes
#     # )
#     print("✅ Done.")

# if __name__=="__main__":
#     main()