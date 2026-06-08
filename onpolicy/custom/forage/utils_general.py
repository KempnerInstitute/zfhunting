import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def flatten_list_of_lists(column):
    """Flatten a column that is a list of lists"""
    return column.apply(
        lambda x: (
            [item for sublist in x for item in sublist] if isinstance(x, list) else x
        )
    )

def cast_list_to_np_array(column):
    """Cast a column that is a list to a numpy array"""
    return column.apply(lambda x: np.array(x) if isinstance(x, list) else x)


def unlist_single_element_lists(column):
    """Un-list columns that contain single-element lists"""
    return column.apply(lambda x: x[0] if isinstance(x, list) and len(x) == 1 else x)


def plot_train_metrics(log_dir):
    try:
        csv_path = os.path.join(log_dir, "train_metrics.csv")
        df = pd.read_csv(csv_path)

        df["rew_smooth"] = df["average_episode_reward"].rolling(20, min_periods=1).mean()

        _, axes = plt.subplots(3, 1, sharex=True, figsize=(6,10))
        axes[0].plot(df["step"], df["rew_smooth"])
        axes[0].set_ylabel("Avg Return (Smoothed)")

        axes[1].plot(df["step"], df["policy_loss"], label="policy")
        axes[1].plot(df["step"], df["value_loss"], label="value")
        axes[1].legend()
        axes[1].set_ylabel("Loss")

        axes[2].plot(df["step"], df["dist_entropy"])
        axes[2].set_ylabel("Entropy")

        axes[2].set_xlabel("Environment Steps")

        plt.tight_layout()
        plt.savefig(os.path.join(log_dir, "training_curve.png"))
    except Exception as e:
        print("Error plotting training metrics:", e)