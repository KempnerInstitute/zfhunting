import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import argparse
import os
from utils_figstyle import set_nature_style
set_nature_style()

# Check if we're in interactive mode or batch mode
batchmode = False
if "ipykernel_launcher" in sys.argv[0]:
    print("Interactive mode")
else:
    batchmode = True
    print("Batch/CLI mode")

default_dir = "/home/raaghav/zfish/onpolicy/custom/forage/results/rmappo-MultiAgentForagingEnv-check/"
if not os.path.exists(default_dir): # Running on cluster
    default_dir = "/n/home04/ramalik/ZFish/onpolicy/custom/forage/results/rmappo-MultiAgentForagingEnv-check"

outputs_folder = "./results/rmappo-MultiAgentForagingEnv-check/20250916_153414_1_bao_vd_0.006_fdr_10_run_3/outputs"

if batchmode:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "outputs_folder",
        default=outputs_folder,
        nargs="?",
    )
    parser.add_argument(
        "--correct_controls",
        action="store_true",
        help="Whether to correct control values for sim FPS",
        default=False,
    )
    args = parser.parse_args()
    outputs_folder = args.outputs_folder

print(f"Using outputs folder: {outputs_folder}")
FPS_SIM = 8

# Read the CSV file
csv_path = os.path.join(outputs_folder, "additional_exps", "all_experiment_data.csv")
if not os.path.exists(csv_path):
    print(f"CSV file not found: {csv_path}")
    raise SystemExit

df = pd.read_csv(csv_path)
print(f"Loaded CSV with {len(df)} rows")

# Create plots directory
plots_dir = os.path.join(outputs_folder, "additional_exps", "plots")
os.makedirs(plots_dir, exist_ok=True)

# Define experiment types and their properties (removed num_bots)
experiments = [
    {
        'name': 'food_speed',
        'control_name': 'control_food_speed',
        'xlabel': 'Food Speed (mm/s)',
        'col_idx': 0
    },
    {
        'name': 'food_density', 
        'control_name': 'control_food_density',
        'xlabel': 'Food Density (count/mm²)',
        'col_idx': 1
    },
    {
        'name': 'limit_convergence',
        'control_name': 'control_limit_convergence', 
        'xlabel': 'Δ Max Vergence (°)',
        'col_idx': 2
    },
    {
        'name': 'limit_divergence',
        'control_name': 'control_limit_divergence',
        'xlabel': 'Δ Min Vergence (°)', 
        'col_idx': 3
    }
]

# Create the main figure with 3 rows and 4 columns
fig, axes = plt.subplots(3, 4, figsize=(20, 15))

# Row labels
row_labels = [
    'Eating Events per Episode',
    'AUC Difference (°)', 
    'Average Success Duration (s)'
]

# First pass: collect all data to determine y-axis ranges
row_data_ranges = [[], [], []]  # One list per row to collect y-values

for exp in experiments:
    exp_data = df[df['experiment_type'] == exp['name']]
    control_data = df[df['experiment_type'] == exp['control_name']]
    
    if len(exp_data) == 0:
        continue
    
    # Extract data for range calculation
    all_eating_values = exp_data['eating_events'].values
    all_eating_errors = exp_data['eating_events_se'].values
    all_auc_diffs = exp_data['auc_difference'].values
    all_auc_errors = exp_data['auc_difference_se'].values
    all_success_durations = exp_data['success_duration'].values
    all_duration_errors = exp_data['success_duration_se'].values
    
    # Add control data if available
    if len(control_data) > 0:
        control_eating = control_data['eating_events'].iloc[0]
        control_eating_se = control_data['eating_events_se'].iloc[0]
        control_auc = control_data['auc_difference'].iloc[0]
        control_auc_se = control_data['auc_difference_se'].iloc[0]
        control_duration = control_data['success_duration'].iloc[0]
        control_duration_se = control_data['success_duration_se'].iloc[0]
        
        if args.correct_controls:
            control_duration *= FPS_SIM
            control_duration_se *= FPS_SIM
    else:
        control_eating = control_eating_se = None
        control_auc = control_auc_se = None
        control_duration = control_duration_se = None
    
    # Collect y-values with error bars for each row
    # Row 0: Eating events
    valid_eating = ~np.isnan(all_eating_values)
    if np.sum(valid_eating) > 0:
        row_data_ranges[0].extend(all_eating_values[valid_eating] + all_eating_errors[valid_eating])
        row_data_ranges[0].extend(all_eating_values[valid_eating] - all_eating_errors[valid_eating])
    if control_eating is not None:
        row_data_ranges[0].extend([control_eating + control_eating_se, control_eating - control_eating_se])
    
    # Row 1: AUC differences
    valid_auc = ~np.isnan(all_auc_diffs)
    if np.sum(valid_auc) > 0:
        row_data_ranges[1].extend(all_auc_diffs[valid_auc] + all_auc_errors[valid_auc])
        row_data_ranges[1].extend(all_auc_diffs[valid_auc] - all_auc_errors[valid_auc])
    if control_auc is not None:
        row_data_ranges[1].extend([control_auc + control_auc_se, control_auc - control_auc_se])
    
    # Row 2: Success durations
    valid_duration = ~np.isnan(all_success_durations)
    if np.sum(valid_duration) > 0:
        row_data_ranges[2].extend(all_success_durations[valid_duration] + all_duration_errors[valid_duration])
        row_data_ranges[2].extend(all_success_durations[valid_duration] - all_duration_errors[valid_duration])
    if control_duration is not None:
        row_data_ranges[2].extend([control_duration + control_duration_se, control_duration - control_duration_se])

# Calculate shared y-limits for each row
shared_ylims = []
for row in range(3):
    if len(row_data_ranges[row]) > 0:
        y_min = np.min(row_data_ranges[row])
        y_max = np.max(row_data_ranges[row])
        # Add 5% padding
        y_range = y_max - y_min
        shared_ylims.append((y_min - 0.05*y_range, y_max + 0.05*y_range))
    else:
        shared_ylims.append((0, 1))  # Default range if no data

# Second pass: create the plots
for exp in experiments:
    exp_data = df[df['experiment_type'] == exp['name']]
    control_data = df[df['experiment_type'] == exp['control_name']]
    
    if len(exp_data) == 0:
        print(f"No data found for {exp['name']}, leaving plots empty")
        col_idx = exp['col_idx']
        # Set up empty plots with proper labels
        for row in range(3):
            ax = axes[row, col_idx]
            ax.set_xlabel(exp['xlabel'], fontsize=20)
            if col_idx == 0:
                ax.set_ylabel(row_labels[row], fontsize=20)
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='both', which='major', labelsize=18)
            ax.text(0.5, 0.5, 'No Data', transform=ax.transAxes, 
                   ha='center', va='center', fontsize=14, alpha=0.5)
            ax.set_ylim(shared_ylims[row])
        continue
        
    print(f"\nProcessing {exp['name']} with {len(exp_data)} data points")
    
    # Extract data
    all_parameter_values = exp_data['parameter_value'].values
    all_eating_values = exp_data['eating_events'].values
    all_eating_errors = exp_data['eating_events_se'].values
    all_auc_diffs = exp_data['auc_difference'].values
    all_auc_errors = exp_data['auc_difference_se'].values
    all_success_durations = exp_data['success_duration'].values
    all_duration_errors = exp_data['success_duration_se'].values
    
    # Get control values
    control_param_value = None
    avg_eating_per_episode = None
    std_eating_per_episode = None
    control_auc_difference = None
    control_auc_se = None
    avg_success_duration = None
    std_success_duration = None
    
    if len(control_data) > 0:
        control_param_value = control_data['parameter_value'].iloc[0]
        avg_eating_per_episode = control_data['eating_events'].iloc[0]
        std_eating_per_episode = control_data['eating_events_se'].iloc[0]
        control_auc_difference = control_data['auc_difference'].iloc[0]
        control_auc_se = control_data['auc_difference_se'].iloc[0]
        avg_success_duration = control_data['success_duration'].iloc[0]
        std_success_duration = control_data['success_duration_se'].iloc[0]
        
        if args.correct_controls:
            avg_success_duration *= FPS_SIM
            std_success_duration *= FPS_SIM
    
    col_idx = exp['col_idx']
    
    # Plot 1: Eating Events (Row 0)
    ax1 = axes[0, col_idx]
    ax1.scatter(all_parameter_values, all_eating_values, alpha=0.6, s=50, c='C0')
    ax1.errorbar(all_parameter_values, all_eating_values, yerr=all_eating_errors, 
                fmt='o', alpha=0.6, capsize=3, capthick=1, color='C0')
    
    # Add regression line
    valid_eating = ~np.isnan(all_eating_values)
    if np.sum(valid_eating) > 1:
        z = np.polyfit(all_parameter_values[valid_eating], all_eating_values[valid_eating], 2)
        p = np.poly1d(z)
        x_smooth = np.linspace(all_parameter_values[valid_eating].min(),
                            all_parameter_values[valid_eating].max(), 100)
        ax1.plot(x_smooth, p(x_smooth), "darkblue", alpha=1.0, linewidth=3)
    
    # Add control point
    if control_param_value is not None:
        ax1.scatter(control_param_value, avg_eating_per_episode, 
                   alpha=0.8, s=100, c='red', marker='s', label='Control', edgecolors='black')
        ax1.errorbar(control_param_value, avg_eating_per_episode, yerr=std_eating_per_episode, 
                    fmt='s', markersize=10, color='red', alpha=0.8, capsize=5, capthick=2)
    
    ax1.set_xlabel(exp['xlabel'], fontsize=20)
    if col_idx == 0:  # Only leftmost plot gets y-label
        ax1.set_ylabel(row_labels[0], fontsize=20)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='both', which='major', labelsize=18)
    ax1.set_ylim(shared_ylims[0])
    if col_idx == 0 and control_param_value is not None:
        ax1.legend(fontsize=12)
    
    # Special handling for food_density x-axis limits
    if exp['name'] == 'food_density':
        min_density = np.min(all_parameter_values)
        max_density = np.max(all_parameter_values)
        ax1.set_xlim(min_density - 0.05*(max_density - min_density), 
                    max_density + 0.05*(max_density - min_density))
    
    # Plot 2: AUC Difference (Row 1)
    ax2 = axes[1, col_idx]
    valid_auc = ~np.isnan(all_auc_diffs)
    ax2.scatter(all_parameter_values[valid_auc], all_auc_diffs[valid_auc], 
               alpha=0.6, s=50, c='C0')
    ax2.errorbar(all_parameter_values[valid_auc], all_auc_diffs[valid_auc], 
                yerr=all_auc_errors[valid_auc], fmt='o', alpha=0.6, capsize=3, capthick=1, color='C0')
    
    # Add regression line
    if np.sum(valid_auc) > 1:
        z = np.polyfit(all_parameter_values[valid_auc], all_auc_diffs[valid_auc], 2)
        p = np.poly1d(z)
        x_smooth = np.linspace(all_parameter_values[valid_auc].min(),
                            all_parameter_values[valid_auc].max(), 100)
        ax2.plot(x_smooth, p(x_smooth), "darkblue", alpha=1.0, linewidth=3)
    
    # Add control point
    if control_param_value is not None:
        ax2.scatter(control_param_value, control_auc_difference, 
                   alpha=0.8, s=100, c='red', marker='s', edgecolors='black')
        ax2.errorbar(control_param_value, control_auc_difference, yerr=control_auc_se, 
                    fmt='s', markersize=10, color='red', alpha=0.8, capsize=5, capthick=2)
    
    ax2.set_xlabel(exp['xlabel'], fontsize=20)
    if col_idx == 0:  # Only leftmost plot gets y-label
        ax2.set_ylabel(row_labels[1], fontsize=20)
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='both', which='major', labelsize=18)
    ax2.set_ylim(shared_ylims[1])
    
    # Special handling for food_density x-axis limits
    if exp['name'] == 'food_density':
        ax2.set_xlim(min_density - 0.05*(max_density - min_density), 
                    max_density + 0.05*(max_density - min_density))
    
    # Plot 3: Success Duration (Row 2)
    ax3 = axes[2, col_idx]
    valid_duration = ~np.isnan(all_success_durations)
    ax3.scatter(all_parameter_values[valid_duration], all_success_durations[valid_duration], 
               alpha=0.6, s=50, c='C0')
    ax3.errorbar(all_parameter_values[valid_duration], all_success_durations[valid_duration], 
                yerr=all_duration_errors[valid_duration], fmt='o', alpha=0.6, capsize=3, capthick=1, color='C0')
    
    # Add regression line
    if np.sum(valid_duration) > 1:
        z = np.polyfit(all_parameter_values[valid_duration], all_success_durations[valid_duration], 2)
        p = np.poly1d(z)
        x_smooth = np.linspace(all_parameter_values[valid_duration].min(),
                            all_parameter_values[valid_duration].max(), 100)
        ax3.plot(x_smooth, p(x_smooth), "darkblue", alpha=1.0, linewidth=3)
    
    # Add control point
    if control_param_value is not None:
        ax3.scatter(control_param_value, avg_success_duration, 
                   alpha=0.8, s=100, c='red', marker='s', edgecolors='black')
        ax3.errorbar(control_param_value, avg_success_duration, yerr=std_success_duration, 
                    fmt='s', markersize=10, color='red', alpha=0.8, capsize=5, capthick=2)
    
    ax3.set_xlabel(exp['xlabel'], fontsize=20)
    if col_idx == 0:  # Only leftmost plot gets y-label
        ax3.set_ylabel(row_labels[2], fontsize=20)
    ax3.grid(True, alpha=0.3)
    ax3.tick_params(axis='both', which='major', labelsize=18)
    ax3.set_ylim(shared_ylims[2])
    
    # Special handling for food_density x-axis limits
    if exp['name'] == 'food_density':
        ax3.set_xlim(min_density - 0.05*(max_density - min_density), 
                    max_density + 0.05*(max_density - min_density))

plt.tight_layout()

# Save the combined plot
plot_path = os.path.join(plots_dir, "combined_episode_level_analysis_3x4.png")
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"Combined episode-level analysis plot saved to: {plot_path}")
plt.show()

print("\nCombined plot generated successfully")
