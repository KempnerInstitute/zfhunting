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

# Process food_speed experiments
food_speed_data = df[df['experiment_type'] == 'food_speed']
control_food_speed_data = df[df['experiment_type'] == 'control_food_speed']

if len(food_speed_data) > 0:
    print(f"\nGenerating food_speed plots with {len(food_speed_data)} data points")
    
    # Extract data
    all_food_speeds = food_speed_data['parameter_value'].values
    all_eating_values = food_speed_data['eating_events'].values
    all_eating_errors = food_speed_data['eating_events_se'].values
    all_auc_diffs = food_speed_data['auc_difference'].values
    all_auc_errors = food_speed_data['auc_difference_se'].values
    all_success_durations = food_speed_data['success_duration'].values
    all_duration_errors = food_speed_data['success_duration_se'].values
    
    # Get control values
    if len(control_food_speed_data) > 0:
        control_food_speed = control_food_speed_data['parameter_value'].iloc[0]
        avg_eating_per_episode = control_food_speed_data['eating_events'].iloc[0]
        std_eating_per_episode = control_food_speed_data['eating_events_se'].iloc[0]
        control_auc_difference = control_food_speed_data['auc_difference'].iloc[0]
        control_auc_se = control_food_speed_data['auc_difference_se'].iloc[0]
        avg_success_duration = control_food_speed_data['success_duration'].iloc[0]
        std_success_duration = control_food_speed_data['success_duration_se'].iloc[0]
        if args.correct_controls:
            avg_success_duration *= FPS_SIM
            std_success_duration *= FPS_SIM

    
    # Create three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))
    
    ax1.scatter(all_food_speeds, all_eating_values, 
                alpha=0.6, s=50, c='C0')
    ax1.errorbar(all_food_speeds, all_eating_values, yerr=all_eating_errors, 
                 fmt='o', alpha=0.6, capsize=3, capthick=1, color='C0')
    # Add regression line
    valid_eating = ~np.isnan(all_eating_values)
    if np.sum(valid_eating) > 1:
        # z = np.polyfit(all_food_speeds[valid_eating], all_eating_values[valid_eating], 1)
        z = np.polyfit(all_food_speeds[valid_eating], all_eating_values[valid_eating], 2)
        p = np.poly1d(z)
        x_smooth = np.linspace(all_food_speeds[valid_eating].min(), 
                          all_food_speeds[valid_eating].max(), 100)
        # fn_label = 'y = '
        # if len(z) == 3:  # 2nd order polynomial
        #     fn_label += f'{z[0]:.1f}x² + {z[1]:.1f}x + {z[2]:.1f}'
        ax1.plot(x_smooth, p(x_smooth), "darkblue", alpha=1.0, linewidth=3)
        # ax1.plot(all_food_speeds[valid_eating], p(all_food_speeds[valid_eating]), "darkblue", alpha=1.0)
    # Add control point for comparison
    if len(control_food_speed_data) > 0:
        ax1.scatter(control_food_speed, avg_eating_per_episode, 
                    alpha=0.8, s=100, c='red', marker='s', label='Control', edgecolors='black')
        ax1.errorbar(control_food_speed, avg_eating_per_episode, yerr=std_eating_per_episode, 
                     fmt='s', markersize=10, color='red', alpha=0.8, capsize=5, capthick=2)
    ax1.set_xlabel('Food Speed (mm/s)', fontsize=20)
    ax1.set_ylabel('Eating Events per Episode', fontsize=20)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=16)
    ax1.tick_params(axis='both', which='major', labelsize=16)
    
    # Plot 2: AUC Difference vs Food Speed
    valid_auc = ~np.isnan(all_auc_diffs)
    ax2.scatter(all_food_speeds[valid_auc], all_auc_diffs[valid_auc], 
                alpha=0.6, s=50, c='C0')
    ax2.errorbar(all_food_speeds[valid_auc], all_auc_diffs[valid_auc], yerr=all_auc_errors[valid_auc], 
                 fmt='o', alpha=0.6, capsize=3, capthick=1, color='C0')
    # Add regression line
    if np.sum(valid_auc) > 1:
        # z = np.polyfit(all_food_speeds[valid_auc], all_auc_diffs[valid_auc], 1)
        z = np.polyfit(all_food_speeds[valid_auc], all_auc_diffs[valid_auc], 2)
        p = np.poly1d(z)
        # ax2.plot(all_food_speeds[valid_auc], p(all_food_speeds[valid_auc]), "darkblue", alpha=1.0)
        x_smooth = np.linspace(all_food_speeds[valid_auc].min(),
                            all_food_speeds[valid_auc].max(), 100)
        ax2.plot(x_smooth, p(x_smooth), "darkblue", alpha=1.0, linewidth=3)
    if len(control_food_speed_data) > 0:
        ax2.scatter(control_food_speed, control_auc_difference, 
                    alpha=0.8, s=100, c='red', marker='s', label='Control', edgecolors='black')
        ax2.errorbar(control_food_speed, control_auc_difference, yerr=control_auc_se, 
                     fmt='s', markersize=10, color='red', alpha=0.8, capsize=5, capthick=2)
    ax2.set_xlabel('Food Speed (mm/s)', fontsize=20)
    ax2.set_ylabel('AUC Difference (°)', fontsize=20)
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='both', which='major', labelsize=16)
    
    # Plot 3: Success Duration vs Food Speed
    valid_duration = ~np.isnan(all_success_durations)
    ax3.scatter(all_food_speeds[valid_duration], all_success_durations[valid_duration], 
                alpha=0.6, s=50, c='C0')
    ax3.errorbar(all_food_speeds[valid_duration], all_success_durations[valid_duration], yerr=all_duration_errors[valid_duration], 
                 fmt='o', alpha=0.6, capsize=3, capthick=1, color='C0')
    # Add regression line
    if np.sum(valid_duration) > 1:
        # z = np.polyfit(all_food_speeds[valid_duration], all_success_durations[valid_duration], 1)
        z = np.polyfit(all_food_speeds[valid_duration], all_success_durations[valid_duration], 2)
        p = np.poly1d(z)
        # ax3.plot(all_food_speeds[valid_duration], p(all_food_speeds[valid_duration]), "darkblue", alpha=1.0)
        x_smooth = np.linspace(all_food_speeds[valid_duration].min(),
                            all_food_speeds[valid_duration].max(), 100)
        ax3.plot(x_smooth, p(x_smooth), "darkblue", alpha=1.0, linewidth=3)
    if len(control_food_speed_data) > 0:
        ax3.scatter(control_food_speed, avg_success_duration, 
                    alpha=0.8, s=100, c='red', marker='s', label='Control', edgecolors='black')
        ax3.errorbar(control_food_speed, avg_success_duration, yerr=std_success_duration, 
                     fmt='s', markersize=10, color='red', alpha=0.8, capsize=5, capthick=2)
    ax3.set_xlabel('Food Speed (mm/s)', fontsize=20)
    ax3.set_ylabel('Average Success Duration (s)', fontsize=20)
    ax3.grid(True, alpha=0.3)
    ax3.tick_params(axis='both', which='major', labelsize=16)
    
    plt.tight_layout()
    
    # Save the plot
    plot_path = os.path.join(plots_dir, "episode_level_food_speed_analysis.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Episode-level analysis plot saved to: {plot_path}")
    plt.show()

# Process food_density experiments
food_density_data = df[df['experiment_type'] == 'food_density']
control_food_density_data = df[df['experiment_type'] == 'control_food_density']

if len(food_density_data) > 0:
    print(f"\nGenerating food_density plots with {len(food_density_data)} data points")
    
    # Extract data
    all_food_densities = food_density_data['parameter_value'].values
    all_eating_values = food_density_data['eating_events'].values
    all_eating_errors = food_density_data['eating_events_se'].values
    all_auc_diffs = food_density_data['auc_difference'].values
    all_auc_errors = food_density_data['auc_difference_se'].values
    all_success_durations = food_density_data['success_duration'].values
    all_duration_errors = food_density_data['success_duration_se'].values
    
    # Get control values
    if len(control_food_density_data) > 0:
        control_food_density = control_food_density_data['parameter_value'].iloc[0]
        avg_eating_per_episode = control_food_density_data['eating_events'].iloc[0]
        std_eating_per_episode = control_food_density_data['eating_events_se'].iloc[0]
        control_auc_difference = control_food_density_data['auc_difference'].iloc[0]
        control_auc_se = control_food_density_data['auc_difference_se'].iloc[0]
        avg_success_duration = control_food_density_data['success_duration'].iloc[0]
        std_success_duration = control_food_density_data['success_duration_se'].iloc[0]

        if args.correct_controls:
            avg_success_duration *= FPS_SIM
            std_success_duration *= FPS_SIM
    
    # Create three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))

    min_density = np.min(all_food_densities)
    max_density = np.max(all_food_densities)
    
    ax1.scatter(all_food_densities, all_eating_values, 
                alpha=0.6, s=50, c='C0')
    ax1.errorbar(all_food_densities, all_eating_values, yerr=all_eating_errors, 
                 fmt='o', alpha=0.6, capsize=3, capthick=1, color='C0')
    # Add regression line
    valid_eating = ~np.isnan(all_eating_values)
    if np.sum(valid_eating) > 1:
        # z = np.polyfit(all_food_densities[valid_eating], all_eating_values[valid_eating], 1)
        z = np.polyfit(all_food_densities[valid_eating], all_eating_values[valid_eating], 2)
        p = np.poly1d(z)
        # ax1.plot(all_food_densities[valid_eating], p(all_food_densities[valid_eating]), "darkblue", alpha=1.0)
        x_smooth = np.linspace(all_food_densities[valid_eating].min(),
                            all_food_densities[valid_eating].max(), 100)
        ax1.plot(x_smooth, p(x_smooth), "darkblue", alpha=1.0, linewidth=3)
    # Add control point for comparison
    if len(control_food_density_data) > 0:
        ax1.scatter(control_food_density, avg_eating_per_episode, 
                    alpha=0.8, s=100, c='red', marker='s', label='Control', edgecolors='black')
        ax1.errorbar(control_food_density, avg_eating_per_episode, yerr=std_eating_per_episode, 
                        fmt='s', markersize=10, color='red', alpha=0.8, capsize=5, capthick=2)
    ax1.set_xlabel('Food Density (count/mm²)', fontsize=20)
    ax1.set_ylabel('Eating Events per Episode', fontsize=20)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=16)
    ax1.tick_params(axis='both', which='major', labelsize=16)
    ax1.set_xlim(min_density - 0.05*(max_density - min_density), max_density + 0.05*(max_density - min_density))
    
    # Plot 2: AUC Difference vs Food Density
    valid_auc = ~np.isnan(all_auc_diffs)
    ax2.scatter(all_food_densities[valid_auc], all_auc_diffs[valid_auc], 
                alpha=0.6, s=50, c='C0', label='Food Density Experiments')
    ax2.errorbar(all_food_densities[valid_auc], all_auc_diffs[valid_auc], yerr=all_auc_errors[valid_auc], 
                 fmt='o', alpha=0.6, capsize=3, capthick=1, color='C0')
    # Add regression line
    if np.sum(valid_auc) > 1:
        # z = np.polyfit(all_food_densities[valid_auc], all_auc_diffs[valid_auc], 1)
        z = np.polyfit(all_food_densities[valid_auc], all_auc_diffs[valid_auc], 2)
        p = np.poly1d(z)
        # ax2.plot(all_food_densities[valid_auc], p(all_food_densities[valid_auc]), "darkblue", alpha=1.0)
        x_smooth = np.linspace(all_food_densities[valid_auc].min(),
                            all_food_densities[valid_auc].max(), 100)
        ax2.plot(x_smooth, p(x_smooth), "darkblue", alpha=1.0, linewidth=3)
    if len(control_food_density_data) > 0:
        ax2.scatter(control_food_density, control_auc_difference, 
                    alpha=0.8, s=100, c='red', marker='s', label='Control', edgecolors='black')
        ax2.errorbar(control_food_density, control_auc_difference, yerr=control_auc_se, 
                        fmt='s', markersize=10, color='red', alpha=0.8, capsize=5, capthick=2)
    ax2.set_xlabel('Food Density (count/mm²)', fontsize=20)
    ax2.set_ylabel('AUC Difference (°)', fontsize=20)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(min_density - 0.05*(max_density - min_density), max_density + 0.05*(max_density - min_density))
    ax2.tick_params(axis='both', which='major', labelsize=16)
    
    # Plot 3: Success Duration vs Food Density
    valid_duration = ~np.isnan(all_success_durations)
    ax3.scatter(all_food_densities[valid_duration], all_success_durations[valid_duration], 
                alpha=0.6, s=50, c='C0', label='Food Density Experiments')
    ax3.errorbar(all_food_densities[valid_duration], all_success_durations[valid_duration], yerr=all_duration_errors[valid_duration], 
                 fmt='o', alpha=0.6, capsize=3, capthick=1, color='C0')
    # Add regression line
    if np.sum(valid_duration) > 1:
        # z = np.polyfit(all_food_densities[valid_duration], all_success_durations[valid_duration], 1)
        z = np.polyfit(all_food_densities[valid_duration], all_success_durations[valid_duration], 2)
        p = np.poly1d(z)
        # ax3.plot(all_food_densities[valid_duration], p(all_food_densities[valid_duration]), "darkblue", alpha=1.0)
        x_smooth = np.linspace(all_food_densities[valid_duration].min(),
                            all_food_densities[valid_duration].max(), 100)
        ax3.plot(x_smooth, p(x_smooth), "darkblue", alpha=1.0, linewidth=3)

    if len(control_food_density_data) > 0:
        ax3.scatter(control_food_density, avg_success_duration, 
                    alpha=0.8, s=100, c='red', marker='s', label='Control', edgecolors='black')
        ax3.errorbar(control_food_density, avg_success_duration, yerr=std_success_duration, 
                        fmt='s', markersize=10, color='red', alpha=0.8, capsize=5, capthick=2)
    ax3.set_xlabel('Food Density (count/mm²)', fontsize=20)
    ax3.set_ylabel('Average Success Duration (s)', fontsize=20)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(min_density - 0.05*(max_density - min_density), max_density + 0.05*(max_density - min_density))
    ax3.tick_params(axis='both', which='major', labelsize=16)
    
    plt.tight_layout()
    
    # Save the plot
    plot_path = os.path.join(plots_dir, "episode_level_food_density_analysis.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Episode-level analysis plot saved to: {plot_path}")
    plt.show()

# Process limit_convergence experiments
limit_convergence_data = df[df['experiment_type'] == 'limit_convergence']
control_limit_convergence_data = df[df['experiment_type'] == 'control_limit_convergence']

if len(limit_convergence_data) > 0:
    print(f"\nGenerating limit_convergence plots with {len(limit_convergence_data)} data points")
    
    # Extract data
    all_limit_convergences = limit_convergence_data['parameter_value'].values
    all_eating_values = limit_convergence_data['eating_events'].values
    all_eating_errors = limit_convergence_data['eating_events_se'].values
    all_auc_diffs = limit_convergence_data['auc_difference'].values
    all_auc_errors = limit_convergence_data['auc_difference_se'].values
    all_success_durations = limit_convergence_data['success_duration'].values
    all_duration_errors = limit_convergence_data['success_duration_se'].values
    
    # Get control values
    if len(control_limit_convergence_data) > 0:
        control_limit_convergence = control_limit_convergence_data['parameter_value'].iloc[0]
        avg_eating_per_episode = control_limit_convergence_data['eating_events'].iloc[0]
        std_eating_per_episode = control_limit_convergence_data['eating_events_se'].iloc[0]
        control_auc_difference = control_limit_convergence_data['auc_difference'].iloc[0]
        control_auc_se = control_limit_convergence_data['auc_difference_se'].iloc[0]
        avg_success_duration = control_limit_convergence_data['success_duration'].iloc[0]
        std_success_duration = control_limit_convergence_data['success_duration_se'].iloc[0]

        if args.correct_controls:
            avg_success_duration *= FPS_SIM
            std_success_duration *= FPS_SIM
    
    # Create three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))

    ax1.scatter(all_limit_convergences, all_eating_values, 
                alpha=0.6, s=50, c='C0')
    ax1.errorbar(all_limit_convergences, all_eating_values, yerr=all_eating_errors, 
                 fmt='o', alpha=0.6, capsize=3, capthick=1, color='C0')
    # Add regression line
    valid_eating = ~np.isnan(all_eating_values)
    if np.sum(valid_eating) > 1:
        # z = np.polyfit(all_limit_convergences[valid_eating], all_eating_values[valid_eating], 1)
        z = np.polyfit(all_limit_convergences[valid_eating], all_eating_values[valid_eating], 2)
        p = np.poly1d(z)
        # ax1.plot(all_limit_convergences[valid_eating], p(all_limit_convergences[valid_eating]), "darkblue", alpha=1.0)
        x_smooth = np.linspace(all_limit_convergences[valid_eating].min(),
                            all_limit_convergences[valid_eating].max(), 100)
        ax1.plot(x_smooth, p(x_smooth), "darkblue", alpha=1.0, linewidth=3)
    # Add control point for comparison
    if len(control_limit_convergence_data) > 0:
        ax1.scatter(control_limit_convergence, avg_eating_per_episode, 
                    alpha=0.8, s=100, c='red', marker='s', label='Control', edgecolors='black')
        ax1.errorbar(control_limit_convergence, avg_eating_per_episode, yerr=std_eating_per_episode, 
                        fmt='s', markersize=10, color='red', alpha=0.8, capsize=5, capthick=2)
    ax1.set_xlabel('Δ Max Vergence (°)', fontsize=20)
    ax1.set_ylabel('Eating Events per Episode', fontsize=20)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=16)
    ax1.tick_params(axis='both', which='major', labelsize=16)
    
    # Plot 2: AUC Difference vs Limit Convergence
    valid_auc = ~np.isnan(all_auc_diffs)
    ax2.scatter(all_limit_convergences[valid_auc], all_auc_diffs[valid_auc], 
                alpha=0.6, s=50, c='C0', label='Limit Convergence Experiments')
    ax2.errorbar(all_limit_convergences[valid_auc], all_auc_diffs[valid_auc], yerr=all_auc_errors[valid_auc], 
                 fmt='o', alpha=0.6, capsize=3, capthick=1, color='C0')
    # Add regression line
    if np.sum(valid_auc) > 1:
        # z = np.polyfit(all_limit_convergences[valid_auc], all_auc_diffs[valid_auc], 1)
        z = np.polyfit(all_limit_convergences[valid_auc], all_auc_diffs[valid_auc], 2)
        p = np.poly1d(z)
        # ax2.plot(all_limit_convergences[valid_auc], p(all_limit_convergences[valid_auc]), "darkblue", alpha=1.0)
        x_smooth = np.linspace(all_limit_convergences[valid_auc].min(),
                            all_limit_convergences[valid_auc].max(), 100)
        ax2.plot(x_smooth, p(x_smooth), "darkblue", alpha=1.0, linewidth=3)
    if len(control_limit_convergence_data) > 0:
        ax2.scatter(control_limit_convergence, control_auc_difference, 
                    alpha=0.8, s=100, c='red', marker='s', label='Control', edgecolors='black')
        ax2.errorbar(control_limit_convergence, control_auc_difference, yerr=control_auc_se, 
                        fmt='s', markersize=10, color='red', alpha=0.8, capsize=5, capthick=2)
    ax2.set_xlabel('Δ Max Vergence (°)', fontsize=20)
    ax2.set_ylabel('AUC Difference (°)', fontsize=20)
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='both', which='major', labelsize=16)
    
    # Plot 3: Success Duration vs Limit Convergence
    valid_duration = ~np.isnan(all_success_durations)
    ax3.scatter(all_limit_convergences[valid_duration], all_success_durations[valid_duration], 
                alpha=0.6, s=50, c='C0', label='Limit Convergence Experiments')
    ax3.errorbar(all_limit_convergences[valid_duration], all_success_durations[valid_duration], yerr=all_duration_errors[valid_duration], 
                 fmt='o', alpha=0.6, capsize=3, capthick=1, color='C0')
    # Add regression line
    if np.sum(valid_duration) > 1:
        # z = np.polyfit(all_limit_convergences[valid_duration], all_success_durations[valid_duration], 1)
        z = np.polyfit(all_limit_convergences[valid_duration], all_success_durations[valid_duration], 2)
        p = np.poly1d(z)
        # ax3.plot(all_limit_convergences[valid_duration], p(all_limit_convergences[valid_duration]), "darkblue", alpha=1.0)
        x_smooth = np.linspace(all_limit_convergences[valid_duration].min(),
                            all_limit_convergences[valid_duration].max(), 100)
        ax3.plot(x_smooth, p(x_smooth), "darkblue", alpha=1.0, linewidth=3)
    if len(control_limit_convergence_data) > 0:
        ax3.scatter(control_limit_convergence, avg_success_duration, 
                    alpha=0.8, s=100, c='red', marker='s', label='Control', edgecolors='black')
        ax3.errorbar(control_limit_convergence, avg_success_duration, yerr=std_success_duration, 
                        fmt='s', markersize=10, color='red', alpha=0.8, capsize=5, capthick=2)
    ax3.set_xlabel('Δ Max Vergence (°)', fontsize=20)
    ax3.set_ylabel('Average Success Duration (s)', fontsize=20)
    ax3.grid(True, alpha=0.3)
    ax3.tick_params(axis='both', which='major', labelsize=16)
    
    plt.tight_layout()
    
    # Save the plot
    plot_path = os.path.join(plots_dir, "episode_level_limit_convergence_analysis.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Episode-level analysis plot saved to: {plot_path}")
    plt.show()

# Process limit_divergence experiments
limit_divergence_data = df[df['experiment_type'] == 'limit_divergence']
control_limit_divergence_data = df[df['experiment_type'] == 'control_limit_divergence']

if len(limit_divergence_data) > 0:
    print(f"\nGenerating limit_divergence plots with {len(limit_divergence_data)} data points")
    
    # Extract data
    all_limit_divergences = limit_divergence_data['parameter_value'].values
    all_eating_values = limit_divergence_data['eating_events'].values
    all_eating_errors = limit_divergence_data['eating_events_se'].values
    all_auc_diffs = limit_divergence_data['auc_difference'].values
    all_auc_errors = limit_divergence_data['auc_difference_se'].values
    all_success_durations = limit_divergence_data['success_duration'].values
    all_duration_errors = limit_divergence_data['success_duration_se'].values
    
    # Get control values
    if len(control_limit_divergence_data) > 0:
        control_limit_divergence = control_limit_divergence_data['parameter_value'].iloc[0]
        avg_eating_per_episode = control_limit_divergence_data['eating_events'].iloc[0]
        std_eating_per_episode = control_limit_divergence_data['eating_events_se'].iloc[0]
        control_auc_difference = control_limit_divergence_data['auc_difference'].iloc[0]
        control_auc_se = control_limit_divergence_data['auc_difference_se'].iloc[0]
        avg_success_duration = control_limit_divergence_data['success_duration'].iloc[0]
        std_success_duration = control_limit_divergence_data['success_duration_se'].iloc[0]

        if args.correct_controls:
            avg_success_duration *= FPS_SIM
            std_success_duration *= FPS_SIM
    
    # Create three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))
    
    ax1.scatter(all_limit_divergences, all_eating_values, 
                alpha=0.6, s=50, c='C0')
    ax1.errorbar(all_limit_divergences, all_eating_values, yerr=all_eating_errors, 
                 fmt='o', alpha=0.6, capsize=3, capthick=1, color='C0')
    # Add regression line
    valid_eating = ~np.isnan(all_eating_values)
    if np.sum(valid_eating) > 1:
        # z = np.polyfit(all_limit_divergences[valid_eating], all_eating_values[valid_eating], 1)
        z = np.polyfit(all_limit_divergences[valid_eating], all_eating_values[valid_eating], 2)
        p = np.poly1d(z)
        # ax1.plot(all_limit_divergences[valid_eating], p(all_limit_divergences[valid_eating]), "darkblue", alpha=1.0)
        x_smooth = np.linspace(all_limit_divergences[valid_eating].min(),
                            all_limit_divergences[valid_eating].max(), 100)
        ax1.plot(x_smooth, p(x_smooth), "darkblue", alpha=1.0, linewidth=3)
    # Add control point for comparison
    if len(control_limit_divergence_data) > 0:
        ax1.scatter(control_limit_divergence, avg_eating_per_episode, 
                    alpha=0.8, s=100, c='red', marker='s', label='Control', edgecolors='black')
        ax1.errorbar(control_limit_divergence, avg_eating_per_episode, yerr=std_eating_per_episode, 
                        fmt='s', markersize=10, color='red', alpha=0.8, capsize=5, capthick=2)
    ax1.set_xlabel('Δ Min Vergence (°)', fontsize=20)
    ax1.set_ylabel('Eating Events per Episode', fontsize=20)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=16)
    ax1.tick_params(axis='both', which='major', labelsize=16)
    
    # Plot 2: AUC Difference vs Limit Divergence
    valid_auc = ~np.isnan(all_auc_diffs)
    ax2.scatter(all_limit_divergences[valid_auc], all_auc_diffs[valid_auc], 
                alpha=0.6, s=50, c='C0')
    ax2.errorbar(all_limit_divergences[valid_auc], all_auc_diffs[valid_auc], yerr=all_auc_errors[valid_auc], 
                 fmt='o', alpha=0.6, capsize=3, capthick=1, color='C0')
    # Add regression line
    if np.sum(valid_auc) > 1:
        # z = np.polyfit(all_limit_divergences[valid_auc], all_auc_diffs[valid_auc], 1)
        z = np.polyfit(all_limit_divergences[valid_auc], all_auc_diffs[valid_auc], 2)
        p = np.poly1d(z)
        # ax2.plot(all_limit_divergences[valid_auc], p(all_limit_divergences[valid_auc]), "darkblue", alpha=1.0)
        x_smooth = np.linspace(all_limit_divergences[valid_auc].min(),
                            all_limit_divergences[valid_auc].max(), 100)
        ax2.plot(x_smooth, p(x_smooth), "darkblue", alpha=1.0, linewidth=3)
    if len(control_limit_divergence_data) > 0:
        ax2.scatter(control_limit_divergence, control_auc_difference, 
                    alpha=0.8, s=100, c='red', marker='s', label='Control', edgecolors='black')
        ax2.errorbar(control_limit_divergence, control_auc_difference, yerr=control_auc_se, 
                        fmt='s', markersize=10, color='red', alpha=0.8, capsize=5, capthick=2)
    ax2.set_xlabel('Δ Min Vergence (°)', fontsize=20)
    ax2.set_ylabel('AUC Difference (°)', fontsize=20)
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='both', which='major', labelsize=16)
    
    # Plot 3: Success Duration vs Limit Divergence
    valid_duration = ~np.isnan(all_success_durations)
    ax3.scatter(all_limit_divergences[valid_duration], all_success_durations[valid_duration], 
                alpha=0.6, s=50, c='C0')
    ax3.errorbar(all_limit_divergences[valid_duration], all_success_durations[valid_duration], yerr=all_duration_errors[valid_duration], 
                 fmt='o', alpha=0.6, capsize=3, capthick=1, color='C0')
    # Add regression line
    if np.sum(valid_duration) > 1:
        # z = np.polyfit(all_limit_divergences[valid_duration], all_success_durations[valid_duration], 1)
        z = np.polyfit(all_limit_divergences[valid_duration], all_success_durations[valid_duration], 2)
        p = np.poly1d(z)
        # ax3.plot(all_limit_divergences[valid_duration], p(all_limit_divergences[valid_duration]), "darkblue", alpha=1.0)
        x_smooth = np.linspace(all_limit_divergences[valid_duration].min(),
                            all_limit_divergences[valid_duration].max(), 100)
        ax3.plot(x_smooth, p(x_smooth), "darkblue", alpha=1.0, linewidth=3)

    if len(control_limit_divergence_data) > 0:
        ax3.scatter(control_limit_divergence, avg_success_duration, 
                    alpha=0.8, s=100, c='red', marker='s', label='Control', edgecolors='black')
        ax3.errorbar(control_limit_divergence, avg_success_duration, yerr=std_success_duration, 
                        fmt='s', markersize=10, color='red', alpha=0.8, capsize=5, capthick=2)
    ax3.set_xlabel('Δ Min Vergence (°)', fontsize=20)
    ax3.set_ylabel('Average Success Duration (s)', fontsize=20)
    ax3.grid(True, alpha=0.3)
    ax3.tick_params(axis='both', which='major', labelsize=16)
    
    plt.tight_layout()
    
    # Save the plot
    plot_path = os.path.join(plots_dir, "episode_level_limit_divergence_analysis.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Episode-level analysis plot saved to: {plot_path}")
    plt.show()

    # Process num_bots experiments
    num_bots_data = df[df['experiment_type'] == 'num_bots']
    control_num_bots_data = df[df['experiment_type'] == 'control_num_bots']

    if len(num_bots_data) > 0:
        print(f"\nGenerating num_bots plots with {len(num_bots_data)} data points")
        
        # Extract data
        all_num_bots = num_bots_data['parameter_value'].values
        all_eating_values = num_bots_data['eating_events'].values
        all_eating_errors = num_bots_data['eating_events_se'].values
        all_auc_diffs = num_bots_data['auc_difference'].values
        all_auc_errors = num_bots_data['auc_difference_se'].values
        all_success_durations = num_bots_data['success_duration'].values
        all_duration_errors = num_bots_data['success_duration_se'].values
        
        # Get control values
        if len(control_num_bots_data) > 0:
            control_num_bots = control_num_bots_data['parameter_value'].iloc[0]
            avg_eating_per_episode = control_num_bots_data['eating_events'].iloc[0]
            std_eating_per_episode = control_num_bots_data['eating_events_se'].iloc[0]
            control_auc_difference = control_num_bots_data['auc_difference'].iloc[0]
            control_auc_se = control_num_bots_data['auc_difference_se'].iloc[0]
            avg_success_duration = control_num_bots_data['success_duration'].iloc[0]
            std_success_duration = control_num_bots_data['success_duration_se'].iloc[0]

            if args.correct_controls:
                avg_success_duration *= FPS_SIM
                std_success_duration *= FPS_SIM
        
        # Create three subplots
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))
        
        ax1.scatter(all_num_bots, all_eating_values, 
                    alpha=0.6, s=50, c='C0')
        ax1.errorbar(all_num_bots, all_eating_values, yerr=all_eating_errors, 
                     fmt='o', alpha=0.6, capsize=3, capthick=1, color='C0')
        # Add regression line
        valid_eating = ~np.isnan(all_eating_values)
        if np.sum(valid_eating) > 1:
            z = np.polyfit(all_num_bots[valid_eating], all_eating_values[valid_eating], 1)
            z = np.polyfit(all_num_bots[valid_eating], all_eating_values[valid_eating], 2)
            p = np.poly1d(z)
            # ax1.plot(all_num_bots[valid_eating], p(all_num_bots[valid_eating]), "darkblue", alpha=1.0)
            x_smooth = np.linspace(all_num_bots[valid_eating].min(),
                                all_num_bots[valid_eating].max(), 100)
            ax1.plot(x_smooth, p(x_smooth), "darkblue", alpha=1.0, linewidth=3)
        # Add control point for comparison
        if len(control_num_bots_data) > 0:
            ax1.scatter(control_num_bots, avg_eating_per_episode, 
                        alpha=0.8, s=100, c='red', marker='s', label='Control', edgecolors='black')
            ax1.errorbar(control_num_bots, avg_eating_per_episode, yerr=std_eating_per_episode, 
                            fmt='s', markersize=10, color='red', alpha=0.8, capsize=5, capthick=2)
        ax1.set_xlabel('Number of Bots', fontsize=20)
        ax1.set_ylabel('Eating Events per Episode', fontsize=20)
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=16)
        ax1.tick_params(axis='both', which='major', labelsize=16)
        
        # Plot 2: AUC Difference vs Num Bots
        valid_auc = ~np.isnan(all_auc_diffs)
        ax2.scatter(all_num_bots[valid_auc], all_auc_diffs[valid_auc], 
                    alpha=0.6, s=50, c='C0')
        ax2.errorbar(all_num_bots[valid_auc], all_auc_diffs[valid_auc], yerr=all_auc_errors[valid_auc], 
                     fmt='o', alpha=0.6, capsize=3, capthick=1, color='C0')
        # Add regression line
        if np.sum(valid_auc) > 1:
            z = np.polyfit(all_num_bots[valid_auc], all_auc_diffs[valid_auc], 1)
            z = np.polyfit(all_num_bots[valid_auc], all_auc_diffs[valid_auc], 2)
            p = np.poly1d(z)
            # ax2.plot(all_num_bots[valid_auc], p(all_num_bots[valid_auc]), "darkblue", alpha=1.0)
            x_smooth = np.linspace(all_num_bots[valid_auc].min(),
                                all_num_bots[valid_auc].max(), 100)
            ax2.plot(x_smooth, p(x_smooth), "darkblue", alpha=1.0, linewidth=3)

        if len(control_num_bots_data) > 0:
            ax2.scatter(control_num_bots, control_auc_difference, 
                        alpha=0.8, s=100, c='red', marker='s', label='Control', edgecolors='black')
            ax2.errorbar(control_num_bots, control_auc_difference, yerr=control_auc_se, 
                            fmt='s', markersize=10, color='red', alpha=0.8, capsize=5, capthick=2)
        ax2.set_xlabel('Number of Bots', fontsize=20)
        ax2.set_ylabel('AUC Difference (°)', fontsize=20)
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis='both', which='major', labelsize=16)
        
        # Plot 3: Success Duration vs Num Bots
        valid_duration = ~np.isnan(all_success_durations)
        ax3.scatter(all_num_bots[valid_duration], all_success_durations[valid_duration], 
                    alpha=0.6, s=50, c='C0')
        ax3.errorbar(all_num_bots[valid_duration], all_success_durations[valid_duration], yerr=all_duration_errors[valid_duration], 
                     fmt='o', alpha=0.6, capsize=3, capthick=1, color='C0')
        # Add regression line
        if np.sum(valid_duration) > 1:
            z = np.polyfit(all_num_bots[valid_duration], all_success_durations[valid_duration], 1)
            z = np.polyfit(all_num_bots[valid_duration], all_success_durations[valid_duration], 2)
            p = np.poly1d(z)
            # ax3.plot(all_num_bots[valid_duration], p(all_num_bots[valid_duration]), "darkblue", alpha=1.0)
            x_smooth = np.linspace(all_num_bots[valid_duration].min(),
                                all_num_bots[valid_duration].max(), 100)
            ax3.plot(x_smooth, p(x_smooth), "darkblue", alpha=1.0, linewidth=3)
        if len(control_num_bots_data) > 0:
            ax3.scatter(control_num_bots, avg_success_duration, 
                        alpha=0.8, s=100, c='red', marker='s', label='Control', edgecolors='black')
            ax3.errorbar(control_num_bots, avg_success_duration, yerr=std_success_duration, 
                            fmt='s', markersize=10, color='red', alpha=0.8, capsize=5, capthick=2)
        ax3.set_xlabel('Number of Bots', fontsize=20)
        ax3.set_ylabel('Average Success Duration (s)', fontsize=20)
        ax3.grid(True, alpha=0.3)
        ax3.tick_params(axis='both', which='major', labelsize=16)
        
        plt.tight_layout()
        
        # Save the plot
        plot_path = os.path.join(plots_dir, "episode_level_num_bots_analysis.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Episode-level analysis plot saved to: {plot_path}")
        plt.show()

print("\nAll plots generated successfully")
