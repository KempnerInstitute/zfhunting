import os
import re
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle

# Now a 2D list: each inner list is AND-matched against folder names to form one plotted series
folder_contains = [
    ['20250812', f'bao_vd_{vd}_fdr_10_run'] for vd in np.arange(-0.003, -0.007, -0.001)
    # add more groups as needed
]

BASE_PATH = "./results/rmappo-MultiAgentForagingEnv-check"

def extract_metrics_from_file(file_path):
    """Extract metrics from performance_metrics.txt file."""
    metrics = {}
    try:
        with open(file_path, 'r') as f:
            content = f.read()

        # Average total episode reward
        m = re.search(r'Average total episode reward:\s*([-+]?\d*\.?\d+)', content)
        if m:
            metrics['reward'] = float(m.group(1))

        # Eating events per episode
        m = re.search(r'Eating events per episode:\s*([-+]?\d*\.?\d+)', content)
        if m:
            metrics['eating_events'] = float(m.group(1))

        # Average AUC (Success)
        m = re.search(r'Average AUC \(Success\):\s*([-+]?\d*\.?\d+)', content)
        if m:
            metrics['auc_success'] = float(m.group(1))

        # Average AUC (Non-Tracking)
        m = re.search(r'Average AUC \(Non-Tracking\):\s*([-+]?\d*\.?\d+)', content)
        if m:
            metrics['auc_non_tracking'] = float(m.group(1))

    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except Exception as e:
        print(f"Error reading {file_path}: {e}")

    return metrics

def collect_group_data(base_path, group_substrings):
    """
    Return a list of metrics dicts for one group (all substrings must be in folder name).
    Adds 'folder' key for provenance.
    """
    group_data = []
    if not os.path.exists(base_path):
        print(f"Base path does not exist: {base_path}")
        return group_data

    for subfolder in os.listdir(base_path):
        subfolder_path = os.path.join(base_path, subfolder)
        if not os.path.isdir(subfolder_path):
            continue

        # AND filter: all substrings must be present in the folder name
        if not all(s in subfolder for s in group_substrings):
            continue

        metrics_file = os.path.join(subfolder_path, 'outputs', 'figures', 'performance_metrics.txt')
        if not os.path.exists(metrics_file):
            continue

        metrics = extract_metrics_from_file(metrics_file)
        # Only keep complete rows
        required = ['reward', 'eating_events', 'auc_success', 'auc_non_tracking']
        if all(k in metrics for k in required):
            metrics['folder'] = subfolder
            group_data.append(metrics)

    return group_data

def print_sorted_summaries(all_groups_data, group_labels):
    # Print by reward
    print("Runs sorted by Average Total Episode Reward (per group):")
    print("=" * 80)
    for label, data in zip(group_labels, all_groups_data):
        if not data:
            print(f"[{label}] No matching folders with valid metrics found.")
            continue
        data_sorted = sorted(data, key=lambda x: x['reward'])
        print(f"[{label}]")
        for i, run in enumerate(data_sorted, 1):
            print(f"{i}. Folder: {run['folder']}")
            print(f"   Average Total Episode Reward: {run['reward']:.4f}")
            print(f"   Eating Events per Episode:    {run['eating_events']:.4f}")
            print(f"   Average AUC (Success):        {run['auc_success']:.4f}")
            print(f"   Average AUC (Non-Tracking):   {run['auc_non_tracking']:.4f}")
        print()

    # Print by eating events
    print("\nRuns sorted by Eating Events per Episode (per group):")
    print("=" * 80)
    for label, data in zip(group_labels, all_groups_data):
        if not data:
            print(f"[{label}] No matching folders with valid metrics found.")
            continue
        data_sorted = sorted(data, key=lambda x: x['eating_events'])
        print(f"[{label}]")
        for i, run in enumerate(data_sorted, 1):
            print(f"{i}. Folder: {run['folder']}")
            print(f"   Eating Events per Episode:    {run['eating_events']:.4f}")
            print(f"   Average Total Episode Reward: {run['reward']:.4f}")
            print(f"   Average AUC (Success):        {run['auc_success']:.4f}")
            print(f"   Average AUC (Non-Tracking):   {run['auc_non_tracking']:.4f}")
        print()

def scatter_overlay(all_groups_data, group_labels, x_key, y_key, title, x_label, y_label, outfile):
    """
    Overlay one scatter series per group on the same axes.
    Also annotate each point with the number following 'run' in its folder name.
    """
    plt.figure(figsize=(10, 6))
    marker_cycle = cycle(['o', 's', '^', 'D', 'P', 'X', '*', 'v', '<', '>'])  # distinct markers per group

    any_points = False
    for data, label in zip(all_groups_data, group_labels):
        if not data:
            continue
        xs = [d[x_key] for d in data]
        if y_key == 'auc_difference':
            ys = [d['auc_success'] - d['auc_non_tracking'] for d in data]
        else:
            ys = [d[y_key] for d in data]

        m = next(marker_cycle)
        plt.scatter(xs, ys, label=label, alpha=0.7, s=50, marker=m)
        any_points = True

        # Annotate with run numbers
        for x_val, y_val, dct in zip(xs, ys, data):
            run_match = re.search(r'run_(\d+)', dct['folder'])
            if run_match:
                run_num = run_match.group(1)
                plt.annotate(
                    run_num,
                    (x_val, y_val),
                    textcoords="offset points",
                    xytext=(5, 5),  # small offset so text doesn't overlap point
                    fontsize=8,
                    alpha=0.8
                )

    if not any_points:
        print(f"No data available to plot for {title} -> {outfile}")
        return

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend(title="Groups", fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outfile, dpi=200)
    print(f"Saved: {outfile}")
    
def make_group_label(group_substrings):
    """Readable legend label from a group’s AND-substrings."""
    return " & ".join(group_substrings)

def main():
    # 1) Collect data for each group
    all_groups_data = []
    group_labels = []
    for group in folder_contains:
        data = collect_group_data(BASE_PATH, group)
        all_groups_data.append(data)
        group_labels.append(make_group_label(group))

    # 2) Print summaries per group
    print_sorted_summaries(all_groups_data, group_labels)

    # 3) Plots:
    #   For x = reward: y in {auc_success, auc_non_tracking, auc_difference}
    scatter_overlay(
        all_groups_data, group_labels,
        x_key='reward', y_key='auc_success',
        title='AUC (Success) vs Average Total Episode Reward',
        x_label='Average Total Episode Reward',
        y_label='Average AUC (Success)',
        outfile='auc_success_vs_reward.png'
    )
    scatter_overlay(
        all_groups_data, group_labels,
        x_key='reward', y_key='auc_non_tracking',
        title='AUC (Non-Tracking) vs Average Total Episode Reward',
        x_label='Average Total Episode Reward',
        y_label='Average AUC (Non-Tracking)',
        outfile='auc_non_tracking_vs_reward.png'
    )
    scatter_overlay(
        all_groups_data, group_labels,
        x_key='reward', y_key='auc_difference',
        title='AUC Difference (Success - Non-Tracking) vs Average Total Episode Reward',
        x_label='Average Total Episode Reward',
        y_label='AUC Difference',
        outfile='auc_difference_vs_reward.png'
    )

    #   For x = eating_events: y in {auc_success, auc_non_tracking, auc_difference}
    scatter_overlay(
        all_groups_data, group_labels,
        x_key='eating_events', y_key='auc_success',
        title='AUC (Success) vs Eating Events per Episode',
        x_label='Eating Events per Episode',
        y_label='Average AUC (Success)',
        outfile='auc_success_vs_eating_events.png'
    )
    scatter_overlay(
        all_groups_data, group_labels,
        x_key='eating_events', y_key='auc_non_tracking',
        title='AUC (Non-Tracking) vs Eating Events per Episode',
        x_label='Eating Events per Episode',
        y_label='Average AUC (Non-Tracking)',
        outfile='auc_non_tracking_vs_eating_events.png'
    )
    scatter_overlay(
        all_groups_data, group_labels,
        x_key='eating_events', y_key='auc_difference',
        title='AUC Difference (Success - Non-Tracking) vs Eating Events per Episode',
        x_label='Eating Events per Episode',
        y_label='AUC Difference',
        outfile='auc_difference_vs_eating_events.png'
    )

if __name__ == "__main__":
    main()
