#### ACTION PREDICTION RELATED UTILS ####

from re import X
import pandas as pd
import numpy as np
import math
from sklearn.decomposition import PCA
import sys
import argparse
import os
import glob
import tqdm

import utils_report as ru
from utils_behavior import (
    bin_agent_size,
    plot_behavior_densities_1d,
)
from utils_features import (
    get_rnn_state_deltas,
    calculate_column_feature_correlations,
    compute_clumpiness_scores_df,
    dynamic_knn_clumpiness,
    knn_distance_analysis,
    calculate_entropy,
    add_attention_entropy_columns,
)
from utils_sensors import get_sensor_indices_from_cfg
from utils_general import grouped_train_test_split

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import rcParams
import seaborn as sns

rcParams["pdf.fonttype"] = 42  # Use Type 42 (TrueType) fonts to save text as text
rcParams["ps.fonttype"] = 42  # For PostScript as well, if needed

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import reciprocal, uniform, randint
from sklearn.inspection import permutation_importance

import utils_decoding as du
from cfg import FEATURE_METADATA, EXCLUDE_FROM_DECODING
from utils_figsaving import _make_fig
from utils_figstyle import set_nature_style
set_nature_style()

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import reciprocal, uniform, randint
from sklearn.inspection import permutation_importance



def setup_data(dff, behavior_mode="foraging", FEATURE_METADATA=FEATURE_METADATA, EXCLUDE_FROM_DECODING=EXCLUDE_FROM_DECODING,):
    dff, decoding_feature_names = du.prepare_features_for_decoding(
        dff,
        normalize=False,  # Shouldn't be
        behavior_mode=behavior_mode,  # this is the most extensive
    )

    regression_features = [
        k
        for k, v in FEATURE_METADATA.items()
        if v.get("feature_type", "unknown")
        in [
            "scalar",
            # "vector", # position_x and position_y are already included in scalar features
            "circular",
            "probability",
            "boolean",
            "count",
        ]
    ]
    regression_features += (
        decoding_feature_names  # from prepare_features_for_decoding() above
    )
    regression_features = list(set(regression_features))  # Remove duplicates

    not_in_feature_metadata = set(regression_features) - set(FEATURE_METADATA.keys())
    if not_in_feature_metadata:
        print(
            f"Warning: The following features are not in FEATURE_METADATA: {not_in_feature_metadata}"
        )

    EXCLUDE_FROM_DECODING = list(EXCLUDE_FROM_DECODING)  # Ensure it's a list
    EXCLUDE_FROM_DECODING += [
        "position_x",
        "position_y",
        "p_eod_future",
        "p_eod_centered",
    ]  # Additonal manual exclusions
    # print("EXCLUDED:", EXCLUDE_FROM_DECODING)
    regression_features = [
        f
        for f in regression_features
        if f in dff.columns
        if f not in (EXCLUDE_FROM_DECODING + list(not_in_feature_metadata))
    ]
    print(f"Using {len(regression_features)} regression features for mode.")
    print(f"Regression features: {regression_features}")

    # event_counter_colnames = [col for col in dff.columns if col.startswith("time_since_")]
    # print("event_counter_colnames:")
    # for col in event_counter_colnames:
    #     print(col)

    # Extract displacement columns if not already present
    if "displacement_ground_x" not in dff.columns:
        dff["displacement_magnitude"] = dff["displacement_ground"].apply(
            lambda x: np.linalg.norm(x)
        )
    regression_features += ["displacement_magnitude",]

    # DEPENDENTS = [
    #     # Actiona
    #     "move_forward",
    #     "turn_angle",
    #     "emit_eod",
    #     "bite_action",
    # ]
    # print("DEPENDENTS:", DEPENDENTS)

    INDEPENDENTS = regression_features
    print("INDEPENDENTS:", INDEPENDENTS)

    return dff, INDEPENDENTS

def shift_and_train_test_split_data(dff, INDEPENDENTS):
    dff["Y_next"] = dff.groupby(["env_id", "episode_index", "agent_id"])["Y"].shift(-1) # Move index back by 1 to predict the next action
    dff = dff.dropna(subset=["Y_next"])

    train_df, test_df = grouped_train_test_split(
        dff,
        group_cols=["env_id", "episode_index", "agent_id"],
        test_size=0.2,
        random_state=42,
    )

    X_train = train_df.loc[:, INDEPENDENTS].fillna(0)
    X_test = test_df.loc[:, INDEPENDENTS].fillna(0)

    # y_train = train_df["Y"].fillna(0)
    # y_test = test_df["Y"].fillna(0)
    y_train = train_df["Y_next"].fillna(0)  # Predict next action
    y_test = test_df["Y_next"].fillna(0)  # Predict next action


    print("X_train.shape:", X_train.shape)
    print("y_train.shape:", y_train.shape)
    print("X_test.shape:", X_test.shape)
    print("y_test.shape:", y_test.shape)

    # print(y_test[:5])
    # print(y_train.dtype)
    # print(y_train.apply(type).value_counts())

    # print("X_train.dtypes:  ", X_train.dtypes)
    return X_train, y_train, X_test, y_test, INDEPENDENTS


### FIXED DEFINITIONS ####
def discretize_turn3(turn_value):
    if turn_value < -0.5:
        return "L"
    elif turn_value > 0.5:
        return "R"
    else:
        return "C"


def discretize_step_size(step_value):
    return "F" if step_value >= 0.5 else "S"


def assign_discrete_behavior_labels(df, action_var="moveturn"):
    if action_var == "moveturn":
        df["turn_angle_discrete"] = df["turn_angle"].apply(discretize_turn3)
        df["move_forward_discrete"] = df["move_forward"].apply(discretize_step_size)
        df["Y"] = (
            df["turn_angle_discrete"] + df["move_forward_discrete"]
        )  # e.g., 'LF', 'RS'
        # df["Y"] = df["turn_angle_discrete"]
        # df["Y"] = df["move_forward_discrete"]  # e.g., 'F', 'S'
        return df
    else:
        raise ValueError(f"Unknown action_var: {action_var}")

#### EQUAL MASS BINNING ####
def discretize_column_equal_mass(series, num_bins=3, labels=None):
    if labels is None:
        labels = [f"bin{i}" for i in range(num_bins)]
    return pd.qcut(series, q=num_bins, labels=labels, duplicates="drop")


def assign_discrete_behavior_labels_equal_mass(df, action_var="moveturn", plot=True, output_dir=None):
    print(f"Assigning discrete behavior labels using {action_var} with equal mass binning.")
    if action_var == "moveturn":
        df["turn_angle_discrete"] = discretize_column_equal_mass(
            df["turn_angle"], num_bins=3, labels=["Left", "Straight", "Right"]
        )
        df["move_forward_discrete"] = discretize_column_equal_mass(
            df["move_forward"], num_bins=2, labels=["Slow", "Fast"]
        )
        # df["move_forward_discrete"] = discretize_column_equal_mass(
        #     df["move_forward"], num_bins=3, labels=["S", "M", "F"]
        # )

        df["Y"] = df["turn_angle_discrete"].astype(str) + df[
            "move_forward_discrete"
        ].astype(str)
        df = df.dropna(subset=["Y"])

        if plot or output_dir:
            for col in [
                "turn_angle_discrete",
                "move_forward_discrete",
                "Y",
            ]:
                plt.figure(figsize=(3, 2.5))
                df[col].value_counts(normalize=True).plot.barh()
                plt.title(f"{col}")
                plt.xlabel("Proportion")
                # plt.ylabel(col)
                plt.xlim(0, 1)
                plt.tight_layout()
                if output_dir:
                    plt.savefig(f"{output_dir}/{col}.png")
                if plot:
                    plt.show()

    elif action_var == "eod":
        df["p_eod_future_discrete"] = discretize_column_equal_mass(
            df["p_eod_future"], num_bins=3, labels=[
                "Low", 
                "Medium", 
                "High", 
                # "Very High"
                ]
        )
        df["Y"] = df["p_eod_future_discrete"].astype(str)
        df = df.dropna(subset=["Y"])
        if plot or output_dir:
            plt.figure(figsize=(3, 2.5))
            df["p_eod_future_discrete"].value_counts(normalize=True).plot.barh()
            plt.title("p_eod_future_discrete")
            plt.xlabel("Proportion")
            plt.xlim(0, 1)
            plt.tight_layout()
            if output_dir:
                plt.savefig(f"{output_dir}/p_eod_future_discrete.png")
            if plot:
                plt.show()
    else:
        raise ValueError(f"Unknown action_var: {action_var}")
    return df

### PREDICTION


def train_assess_classifier(
    X_train, y_train, 
    X_test, y_test,
    outputs_folder=None,
    outfile_base=None,
    action_var="moveturn",
    FEATURE_METADATA=FEATURE_METADATA,
    model_seed=137,
    HYPOPT_ITER=20,
    HYPOPT_CV=3,
    verbose_rf=2,
    n_jobs_rf=8,
    name_mode="short_name",  # Options: "name", "short_name", "
    param_distributions = {"n_estimators": randint(10, 100), "max_depth": randint(2, 20),},
):
    if HYPOPT_ITER <= 0:
        # Without HypOpt
        clf = RandomForestClassifier(n_estimators=30, n_jobs=n_jobs_rf, oob_score=True, random_state=model_seed)
        clf.fit(X_train, y_train)
        best_params = {
            "n_estimators": clf.n_estimators,
            "max_depth": clf.max_depth,
            }
    else:
        model = RandomForestClassifier(
            n_estimators=100,
            # class_weight="balanced",
            n_jobs=n_jobs_rf,
            oob_score=True,
            random_state=model_seed,
        )
        rnd_search_cv = RandomizedSearchCV(
            model,
            param_distributions,
            n_iter=HYPOPT_ITER,
            verbose=verbose_rf,
            cv=HYPOPT_CV,
            random_state=model_seed,
            n_jobs=n_jobs_rf,
        )

        rnd_search_cv.fit(X_train, y_train)
        clf = rnd_search_cv.best_estimator_
        best_params = rnd_search_cv.best_params_
        model_seed = "RandomForestClassifier-{}".format(
            rnd_search_cv.best_estimator_.random_state
        )

    print(f"{model_seed}: params used:{best_params}")

    ## Assess model performance
    clf_acc_train = clf.oob_score_
    print("{}: Accuracy on train data: {:.2f}".format(model_seed, clf_acc_train))
    clf_acc_test = clf.score(X_test, y_test)
    print("{}: Accuracy on test data: {:.2f}".format(model_seed, clf_acc_test))

    # Baseline classifier
    dummy_clf = DummyClassifier(strategy="most_frequent")
    dummy_clf.fit(X_train, y_train)
    dummy_clf_acc_train = dummy_clf.score(X_train, y_train)
    print(
        "{}: Baseline (most frequent) accuracy on train data: {:.2f}".format(
            model_seed, dummy_clf_acc_train
        )
    )
    dummy_clf_acc_test = dummy_clf.score(X_test, y_test)
    print(
        "{}: Baseline (most frequent) accuracy on test data: {:.2f}".format(
            model_seed, dummy_clf_acc_test
        )
    )

    # Assess feature importance
    print("Assessing feature importance using permutation importance...")
    result = permutation_importance(
        clf, X_train, y_train, n_repeats=30, random_state=42, n_jobs=n_jobs_rf,
    )
    sorted_idx = result.importances_mean.argsort()
    permutation_importances = result.importances[sorted_idx].T
    # print(sorted_idx)

    # Plot permutation importances
    if name_mode in ["name", "short_name"]:
        feature_labels = [
            FEATURE_METADATA.get(f, {}).get(name_mode, f)
            for f in X_train.columns[sorted_idx]
        ]
    else:
        feature_labels = [f for f in X_train.columns[sorted_idx]]

    fig, ax = plt.subplots(figsize=(5, 6))
    ax.boxplot(
        permutation_importances,
        vert=False,
        labels=feature_labels,
    )
    ax.set_title("Permutation Importances for {}".format(action_var))
    # ax.set_title("Permutation importances")
    plt.ylabel("Feature")
    plt.xlabel("Permutation importance")
    fig.tight_layout()

    if outfile_base is not None:
        fig_path = outfile_base + f"_perm_importance_{action_var}.png"
        fig.savefig(fig_path, dpi=300)
        print(f"Saved figure to {fig_path}")
    elif outputs_folder is not None:
        fig_path = os.path.join(outputs_folder, f"perm_importance_{action_var}.png")
        fig.savefig(fig_path, dpi=300)
        print(f"Saved figure to {fig_path}")
    else:
        plt.show()

    # Save classifier results to CSV
    classifier_df = {
        "model_seed": model_seed,
        "clf_acc_test": np.around(clf_acc_test, decimals=2),
        "clf_acc_train": np.around(clf_acc_train, decimals=2),
        "dummy_clf_acc_test": np.around(dummy_clf_acc_test, decimals=2),
        "dummy_clf_acc_train": np.around(dummy_clf_acc_train, decimals=2),
        "best_params": str(best_params),
        "action_var": action_var,
        "n_estimators": rnd_search_cv.best_estimator_.n_estimators,
        "max_depth": rnd_search_cv.best_estimator_.max_depth,
        "n_features": X_train.shape[1],
        "n_samples_train": X_train.shape[0],
        "n_samples_test": X_test.shape[0],
        "n_classes": len(np.unique(y_train)),
        "permutation_importances": permutation_importances.tolist(),
        "feature_labels": feature_labels,
    }
    classifier_df = pd.DataFrame([classifier_df])

    if outfile_base is not None:
        out_csv_path = outfile_base + f"classifier_results_{action_var}.csv"
        classifier_df.to_csv(out_csv_path, index=False)
        print(f"Saved classifier results to {out_csv_path}")
    elif outputs_folder is not None:
        out_csv_path = os.path.join(outputs_folder, f"classifier_results_{action_var}.csv")
        classifier_df.to_csv(out_csv_path, index=False)
        print(f"Saved classifier results to {out_csv_path}")

    return classifier_df


def run_predict_action_report(dff, outfile_base):
    dff, INDEPENDENTS = setup_data(dff, behavior_mode="foraging", FEATURE_METADATA=FEATURE_METADATA, EXCLUDE_FROM_DECODING=EXCLUDE_FROM_DECODING,)
    print(f"Data shape after setup: {dff.shape}")
    print(f"Independent features: {INDEPENDENTS}")

    binning_method = ["equal_mass", "predefined"][0]  # Options
    model_seed = 0
    HYPOPT_ITER = 20
    HYPOPT_CV = 3
    name_mode = "name"

    for action_var in ["moveturn", "eod"]:
        print(f"Processing action variable: {action_var}")
        if binning_method == "predefined":
            dff = assign_discrete_behavior_labels(dff)
        if binning_method == "equal_mass":
            dff = assign_discrete_behavior_labels_equal_mass(dff, action_var=action_var)

        X_train, y_train, X_test, y_test, INDEPENDENTS = shift_and_train_test_split_data(dff, INDEPENDENTS)

        classifier_df = train_assess_classifier(
            X_train, y_train, 
            X_test, y_test,
            action_var=action_var,
            param_distributions = {"n_estimators": randint(10, 100), "max_depth": randint(2, 20),},
            FEATURE_METADATA=FEATURE_METADATA,
            model_seed=model_seed,
            HYPOPT_ITER=HYPOPT_ITER,
            HYPOPT_CV=HYPOPT_CV,
            verbose_rf=2,
            n_jobs_rf=20,
            name_mode=name_mode,
            outputs_folder=None,
            outfile_base=outfile_base
        )


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
        help="Task e.g., 'foraging', '2f1p', etc.",
    )
    args = parser.parse_args()

    flat_pkl_file = args.flat_pkl_file
    model_seed = args.model_seed
    name_mode = args.name_mode
    task = args.task

    # Load the data
    dff, outputs_folder, pkl_str = ru.load_flat_pkl_file(flat_pkl_file, task=task)
    print(f"Loaded data from {flat_pkl_file}")
    print(f"Data shape: {dff.shape}")
    print(f"Outputs will be saved to {outputs_folder}")
    if not os.path.exists(outputs_folder):
        os.makedirs(outputs_folder)
        print(f"Created output directory: {outputs_folder}")

    # Setup data for prediction
    dff, INDEPENDENTS = setup_data(dff, behavior_mode="foraging", FEATURE_METADATA=FEATURE_METADATA, EXCLUDE_FROM_DECODING=EXCLUDE_FROM_DECODING,)
    print(f"Data shape after setup: {dff.shape}")
    print(f"Independent features: {INDEPENDENTS}")

    # Precition 
    binning_method = ["equal_mass", "predefined"][0]  # Options
    batchmode = False
    HYPOPT_ITER = 20
    HYPOPT_CV = 3
    if batchmode:
        HYPOPT_ITER = 20
        HYPOPT_CV = 3

    for action_var in ["moveturn", "eod"]:
        print(f"Processing action variable: {action_var}")
        if binning_method == "predefined":
            dff = assign_discrete_behavior_labels(dff)
        if binning_method == "equal_mass":
            dff = assign_discrete_behavior_labels_equal_mass(dff, action_var=action_var)

        X_train, y_train, X_test, y_test, INDEPENDENTS = shift_and_train_test_split_data(dff, INDEPENDENTS)

        classifier_df = train_assess_classifier(
            X_train, y_train, 
            X_test, y_test,
            action_var=action_var,
            param_distributions = {"n_estimators": randint(10, 100), "max_depth": randint(2, 20),},
            FEATURE_METADATA=FEATURE_METADATA,
            model_seed=model_seed,
            HYPOPT_ITER=20,
            HYPOPT_CV=3,
            verbose_rf=2,
            n_jobs_rf=20,
            name_mode=args.name_mode,
            outputs_folder=None,
            outfile_base=None
        )

