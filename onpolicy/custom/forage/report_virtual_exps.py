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
from utils_behavior import *
import cfg
from utils_features import add_hunting

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

default_dir = "/home/raaghav/zfish/onpolicy/custom/forage/results/rmappo-MultiAgentForagingEnv-check/"
if not os.path.exists(default_dir): # Running on cluster
    default_dir = "/n/home04/ramalik/ZFish/onpolicy/custom/forage/results/rmappo-MultiAgentForagingEnv-check"

# outputs_folder = "/srv/marl/raaghav/marl_zfish/rmappo-MultiAgentForagingEnv-1_agent/20250814_153603_1_bao_vd_0.003_fd_10_action_noise_0.1/outputs/"
# outputs_folder = "/home/raaghav/zfish/onpolicy/custom/forage/results/rmappo-MultiAgentForagingEnv-check/20250808_153214_1_bao_efp_0.05_vd_0.002_fd_10/outputs"
# outputs_folder = "/home/raaghav/zfish/onpolicy/custom/forage/results/rmappo-MultiAgentForagingEnv-check/20250808_153214_1_bao_efp_0.05_vd_0.002_fd_10/additional_exps"
outputs_folder = "./results/rmappo-MultiAgentForagingEnv-check/20250916_153414_1_bao_vd_0.006_fdr_10_run_3/outputs"
# outputs_folder = "/n/holylfs06/LABS/krajan_lab/Lab/zfish/sonja_results/results/rmappo-MultiAgentForagingEnv-check/20250918_122018_1_fixed_speed_vd_0.006_fdr_10_run_2_food_0.25_1.0_curric_time_norm_step/outputs"

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

# Get all subfolders in additional_exps
additional_exps_dir = os.path.join(outputs_folder, "additional_exps")
if os.path.exists(additional_exps_dir):
    subfolders = [f for f in os.listdir(additional_exps_dir) if os.path.isdir(os.path.join(additional_exps_dir, f))]
    
    # Dictionary to store dataframes from each subfolder
    dff_dict = {}
    
    for subfolder in subfolders:
        subfolder_path = os.path.join(additional_exps_dir, subfolder)
        try:
            flat_pkl_file = get_latest_flat_pkl_file(subfolder_path)
            dff_dict[subfolder] = pd.read_pickle(flat_pkl_file)
            # Sort the dataframe by the specified columns
            dff_dict[subfolder] = dff_dict[subfolder].sort_values(
                by=["env_id", "episode_index", "agent_id", "time_step"]
            ).reset_index(drop=True)
            print(f"Loaded {subfolder}: {flat_pkl_file}")
            print(f"  Shape: {dff_dict[subfolder].shape}")
        except FileNotFoundError:
            print(f"No .pkl files found in {subfolder}")
    
    print(f"\nLoaded {len(dff_dict)} dataframes from additional_exps subfolders")
else:
    print(f"No additional_exps directory found in {outputs_folder}")
    raise SystemExit

# Load and analyze control dataframe
if 'control' in dff_dict:
    control_df = dff_dict['control']
    print(f"\nAnalyzing control dataframe.")
    print(f"Shape: {control_df.shape}")
    
    # Calculate average eating events per episode
    eating_per_episode = control_df.groupby(['env_id', 'episode_index'])['eating_event'].sum()
    avg_eating_per_episode = eating_per_episode.mean()
    std_eating_per_episode = eating_per_episode.std() / (np.sqrt(len(eating_per_episode)))
    print(f"Eating events: {avg_eating_per_episode:.2f} ± {std_eating_per_episode:.2f} per episode")
    
    # Calculate cumulative reward
    final_rewards_per_episode = control_df.groupby(['env_id', 'episode_index'])['cumulative_reward'].last()
    avg_final_reward = final_rewards_per_episode.mean()
    std_final_reward = final_rewards_per_episode.std() / (np.sqrt(len(final_rewards_per_episode)))
    print(f"Final reward: {avg_final_reward:.2f} ± {std_final_reward:.2f} per episode")
    
    # Calculate AUC difference for each environment separately
    env_auc_differences = []
    
    for env_id in control_df['env_id'].unique():
        env_df = control_df[control_df['env_id'] == env_id]
        tracking_sequences_df = analyze_vergence_during_food_tracking(env_df)
        
        if len(tracking_sequences_df) > 0:
            non_tracking_vergence = non_tracking_data(env_df, tracking_sequences_df)
            success_trajectories, miss_trajectories = calculate_avg_vergence_by_outcome(tracking_sequences_df)
            
            if len(success_trajectories) > 0 and len(non_tracking_vergence) > 0:
                avg_auc, non_tracking_auc = calculate_auc_hunting_no_hunting(success_trajectories, non_tracking_vergence)
                
                # Convert AUC to degrees
                max_vergence = (cfg.FISH_CONSTANTS["max_left_vergence"] - cfg.FISH_CONSTANTS["max_right_vergence"] + cfg.FISH_CONSTANTS["perception_field"]) * 180 / np.pi
                avg_auc_degrees = avg_auc * max_vergence
                non_tracking_auc_degrees = non_tracking_auc * max_vergence
                auc_difference = avg_auc_degrees - non_tracking_auc_degrees
                
                env_auc_differences.append(auc_difference)
    
    # Calculate mean and standard error across environments
    if env_auc_differences:
        control_auc_difference = np.mean(env_auc_differences)
        control_auc_se = np.std(env_auc_differences) / np.sqrt(len(env_auc_differences))
        print(f"AUC difference (tracking - non-tracking): {control_auc_difference:.2f} ± {control_auc_se:.2f} degrees")
    else:
        control_auc_difference = np.nan
        control_auc_se = 0
        print("Could not calculate AUC difference for control")
    
    # Analyze vergence during food tracking (for other statistics)
    tracking_sequences_df = analyze_vergence_during_food_tracking(control_df)
    non_tracking_vergence = non_tracking_data(control_df, tracking_sequences_df)
    success_trajectories, miss_trajectories = calculate_avg_vergence_by_outcome(tracking_sequences_df)
    avg_auc, non_tracking_auc = calculate_auc_hunting_no_hunting(success_trajectories, non_tracking_vergence)
    
    # Get success data for duration analysis
    success_data = tracking_sequences_df[tracking_sequences_df['outcome'] == 'success']
    success_durations = success_data['tracking_duration'].values
    success_durations = success_durations / cfg.ENV_PARAMS["fps_sim"]  # Convert to seconds
    avg_success_duration = np.mean(success_durations)
    std_success_duration = np.std(success_durations) / (np.sqrt(len(success_durations)))
    
    # Calculate vergence statistics
    tracking_vergence = np.array([ang for _, angles in success_trajectories for ang in angles])
    trk = np.abs(tracking_vergence)
    non = np.abs(np.asarray(non_tracking_vergence))
    trk_std = np.std(trk)
    non_std = np.std(non)
    
    print(f"Average successful hunting sequence duration: {avg_success_duration:.2f} ± {std_success_duration:.2f}")
else:
    print("Control dataframe 'control' not found in loaded data")

# Initialize lists to store all plot data
all_plot_data = []

if 'food_speed' in dff_dict:
    food_speed_df = dff_dict['food_speed']
    print(f"\nAnalyzing food_speed dataframe.")
    print(f"Shape: {food_speed_df.shape}")

    # Dictionary to store data by food speed and env_id
    env_data = {}
    
    # Get unique episodes
    episodes = food_speed_df.groupby(['env_id', 'episode_index'])
    
    for (env_id, episode_index), episode_df in episodes:
        # Get food speed for this episode (should be constant within episode)
        if 'food_speed' not in episode_df.columns:
            raise ValueError("food_speed column not found in food_speed dataframe")
        food_speed = episode_df['food_speed'].iloc[0]
        
        # Calculate eating events for this episode
        eating_events = episode_df['eating_event'].sum()
        
        # Calculate vergence analysis for this episode
        tracking_sequences_df = analyze_vergence_during_food_tracking(episode_df)
        
        if len(tracking_sequences_df) > 0:
            non_tracking_vergence = non_tracking_data(episode_df, tracking_sequences_df)
            success_trajectories, miss_trajectories = calculate_avg_vergence_by_outcome(tracking_sequences_df)
            
            if len(success_trajectories) > 0 and len(non_tracking_vergence) > 0:
                avg_auc, non_tracking_auc = calculate_auc_hunting_no_hunting(success_trajectories, non_tracking_vergence)
                
                # Convert AUC to degrees
                max_vergence = (cfg.FISH_CONSTANTS["max_left_vergence"] - cfg.FISH_CONSTANTS["max_right_vergence"] + cfg.FISH_CONSTANTS["perception_field"]) * 180 / np.pi
                avg_auc_degrees = avg_auc * max_vergence
                non_tracking_auc_degrees = non_tracking_auc * max_vergence
                auc_difference = avg_auc_degrees - non_tracking_auc_degrees
                
                # Calculate successful hunting duration
                success_data = tracking_sequences_df[tracking_sequences_df['outcome'] == 'success']
                if len(success_data) > 0:
                    success_duration = success_data['tracking_duration'].mean()
                else:
                    success_duration = np.nan
                
                # Store data by env_id and food_speed
                key = (env_id, food_speed)
                if key not in env_data:
                    env_data[key] = {'eating': [], 'auc_diff': [], 'success_duration': []}
                
                env_data[key]['eating'].append(eating_events)
                env_data[key]['auc_diff'].append(auc_difference)
                env_data[key]['success_duration'].append(success_duration)

    # Calculate averages and standard errors for each environment
    all_food_speeds = []
    all_eating_values = []
    all_eating_errors = []
    all_auc_diffs = []
    all_auc_errors = []
    all_success_durations = []
    all_duration_errors = []
    
    for (env_id, food_speed), data in env_data.items():
        # Calculate means and standard errors
        eating_mean = np.mean(data['eating'])
        eating_se = np.std(data['eating']) / np.sqrt(len(data['eating']))
        
        auc_diffs = [x for x in data['auc_diff'] if not np.isnan(x)]
        auc_mean = np.mean(auc_diffs) if auc_diffs else np.nan
        auc_se = np.std(auc_diffs) / np.sqrt(len(auc_diffs)) if len(auc_diffs) > 0 else 0
        
        durations = [x for x in data['success_duration'] if not np.isnan(x)]
        duration_mean = np.mean(durations) if durations else np.nan
        duration_se = np.std(durations) / np.sqrt(len(durations)) if len(durations) > 0 else 0
        
        all_food_speeds.append(food_speed)
        all_eating_values.append(eating_mean)
        all_eating_errors.append(eating_se)
        all_auc_diffs.append(auc_mean)
        all_auc_errors.append(auc_se)
        all_success_durations.append(duration_mean)
        all_duration_errors.append(duration_se)
        
        # Store data for CSV export
        all_plot_data.append({
            'experiment_type': 'food_speed',
            'parameter_value': food_speed * cfg.ENV_PARAMS["fps_sim"],  # Convert to mm/s
            'parameter_name': 'food_speed',
            'parameter_units': 'mm/s',
            'eating_events': eating_mean,
            'eating_events_se': eating_se,
            'auc_difference': auc_mean,
            'auc_difference_se': auc_se,
            'success_duration': duration_mean * (1/cfg.ENV_PARAMS["fps_sim"]),  # Convert to seconds
            'success_duration_se': duration_se * (1/cfg.ENV_PARAMS["fps_sim"]),
            'env_id': env_id
        })

    # Convert to numpy arrays
    all_food_speeds = np.array(all_food_speeds)
    all_eating_values = np.array(all_eating_values)
    all_eating_errors = np.array(all_eating_errors)
    all_auc_diffs = np.array(all_auc_diffs)
    all_auc_errors = np.array(all_auc_errors)
    all_success_durations = np.array(all_success_durations)
    all_duration_errors = np.array(all_duration_errors)
    
    # Create plots directory
    plots_dir = os.path.join(outputs_folder, "additional_exps", "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Create three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 8))
    
    ax1.scatter(all_food_speeds, all_eating_values, 
                alpha=0.6, s=50, c='blue', label='Food Speed Experiments')
    ax1.errorbar(all_food_speeds, all_eating_values, yerr=all_eating_errors, 
                 fmt='o', alpha=0.6, capsize=3, capthick=1, color='blue')
    # Add regression line
    valid_eating = ~np.isnan(all_eating_values)
    if np.sum(valid_eating) > 1:
        z = np.polyfit(all_food_speeds[valid_eating], all_eating_values[valid_eating], 1)
        p = np.poly1d(z)
        ax1.plot(all_food_speeds[valid_eating], p(all_food_speeds[valid_eating]), "b--", alpha=0.8)
    # Add control point for comparison
    control_food_speed = cfg.ENV_PARAMS["food_speed"]
    if 'control' in dff_dict:
        ax1.scatter(control_food_speed, avg_eating_per_episode, 
                    alpha=0.8, s=100, c='red', marker='s', label='Control', edgecolors='black')
        ax1.errorbar(control_food_speed, avg_eating_per_episode, yerr=std_eating_per_episode, 
                     fmt='s', markersize=10, color='red', alpha=0.8, capsize=5, capthick=2)
    ax1.set_xlabel('Food Speed (mm/s)', fontsize=16)
    ax1.set_ylabel('Eating Events per Episode', fontsize=16)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=14)
    ax1.tick_params(axis='both', which='major', labelsize=14)
    
    # Plot 2: AUC Difference vs Food Speed
    valid_auc = ~np.isnan(all_auc_diffs)
    ax2.scatter(all_food_speeds[valid_auc], all_auc_diffs[valid_auc], 
                alpha=0.6, s=50, c='blue', label='Food Speed Experiments')
    ax2.errorbar(all_food_speeds[valid_auc], all_auc_diffs[valid_auc], yerr=all_auc_errors[valid_auc], 
                 fmt='o', alpha=0.6, capsize=3, capthick=1, color='blue')
    # Add regression line
    if np.sum(valid_auc) > 1:
        z = np.polyfit(all_food_speeds[valid_auc], all_auc_diffs[valid_auc], 1)
        p = np.poly1d(z)
        ax2.plot(all_food_speeds[valid_auc], p(all_food_speeds[valid_auc]), "b--", alpha=0.8)
    if 'control' in dff_dict:
        ax2.scatter(control_food_speed, control_auc_difference, 
                    alpha=0.8, s=100, c='red', marker='s', label='Control', edgecolors='black')
        ax2.errorbar(control_food_speed, control_auc_difference, yerr=control_auc_se, 
                     fmt='s', markersize=10, color='red', alpha=0.8, capsize=5, capthick=2)
    ax2.set_xlabel('Food Speed (mm/s)', fontsize=16)
    ax2.set_ylabel('AUC Difference (°)', fontsize=16)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=14)
    ax2.tick_params(axis='both', which='major', labelsize=14)
    
    # Plot 3: Success Duration vs Food Speed
    valid_duration = ~np.isnan(all_success_durations)
    all_success_durations = all_success_durations / cfg.ENV_PARAMS["fps_sim"]  # Convert to seconds
    ax3.scatter(all_food_speeds[valid_duration], all_success_durations[valid_duration], 
                alpha=0.6, s=50, c='blue', label='Food Speed Experiments')
    ax3.errorbar(all_food_speeds[valid_duration], all_success_durations[valid_duration], yerr=all_duration_errors[valid_duration], 
                 fmt='o', alpha=0.6, capsize=3, capthick=1, color='blue')
    # Add regression line
    if np.sum(valid_duration) > 1:
        z = np.polyfit(all_food_speeds[valid_duration], all_success_durations[valid_duration], 1)
        p = np.poly1d(z)
        ax3.plot(all_food_speeds[valid_duration], p(all_food_speeds[valid_duration]), "b--", alpha=0.8)
    if 'control' in dff_dict:
        ax3.scatter(control_food_speed, avg_success_duration, 
                    alpha=0.8, s=100, c='red', marker='s', label='Control', edgecolors='black')
        ax3.errorbar(control_food_speed, avg_success_duration, yerr=std_success_duration, 
                     fmt='s', markersize=10, color='red', alpha=0.8, capsize=5, capthick=2)
    ax3.set_xlabel('Food Speed (mm/s)', fontsize=16)
    ax3.set_ylabel('Average Success Duration (s)', fontsize=16)
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=14)
    ax3.tick_params(axis='both', which='major', labelsize=14)
    
    plt.tight_layout()
    
    # Save the plot
    plot_path = os.path.join(plots_dir, "episode_level_food_speed_analysis.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\nEpisode-level analysis plot saved to: {plot_path}")
    plt.show()
        
else:
    print("\nNo dataframes found with food_speed in name")


if 'food_density' in dff_dict:
    food_density_df = dff_dict['food_density']
    print(f"\nAnalyzing food_density dataframe.")
    print(f"Shape: {food_density_df.shape}")

    # Dictionary to store data by food density and env_id
    env_data = {}
    
    # Get unique episodes
    episodes = food_density_df.groupby(['env_id', 'episode_index'])
    
    for (env_id, episode_index), episode_df in episodes:
        # Get food density for this episode (should be constant within episode)
        food_density = episode_df['reset_food_density'].iloc[0] if 'reset_food_density' in episode_df.columns else np.nan
        
        # Calculate eating events for this episode
        eating_events = episode_df['eating_event'].sum()
        
        # Calculate vergence analysis for this episode
        tracking_sequences_df = analyze_vergence_during_food_tracking(episode_df)
        
        if len(tracking_sequences_df) > 0:
            non_tracking_vergence = non_tracking_data(episode_df, tracking_sequences_df)
            success_trajectories, miss_trajectories = calculate_avg_vergence_by_outcome(tracking_sequences_df)
            
            if len(success_trajectories) > 0 and len(non_tracking_vergence) > 0:
                avg_auc, non_tracking_auc = calculate_auc_hunting_no_hunting(success_trajectories, non_tracking_vergence)
                
                # Convert AUC to degrees
                max_vergence = (cfg.FISH_CONSTANTS["max_left_vergence"] - cfg.FISH_CONSTANTS["max_right_vergence"] + cfg.FISH_CONSTANTS["perception_field"]) * 180 / np.pi
                avg_auc_degrees = avg_auc * max_vergence
                non_tracking_auc_degrees = non_tracking_auc * max_vergence
                auc_difference = avg_auc_degrees - non_tracking_auc_degrees
                
                # Calculate successful hunting duration
                success_data = tracking_sequences_df[tracking_sequences_df['outcome'] == 'success']
                if len(success_data) > 0:
                    success_duration = success_data['tracking_duration'].mean()
                else:
                    success_duration = np.nan
                
                # Store data by env_id and food_density
                key = (env_id, food_density)
                if key not in env_data:
                    env_data[key] = {'eating': [], 'auc_diff': [], 'success_duration': []}
                
                env_data[key]['eating'].append(eating_events)
                env_data[key]['auc_diff'].append(auc_difference)
                env_data[key]['success_duration'].append(success_duration)

    # Calculate averages and standard errors for each environment
    all_food_densities = []
    all_eating_values = []
    all_eating_errors = []
    all_auc_diffs = []
    all_auc_errors = []
    all_success_durations = []
    all_duration_errors = []
    
    for (env_id, food_density), data in env_data.items():
        # Calculate means and standard errors
        eating_mean = np.mean(data['eating'])
        eating_se = np.std(data['eating']) / np.sqrt(len(data['eating']))
        
        auc_diffs = [x for x in data['auc_diff'] if not np.isnan(x)]
        auc_mean = np.mean(auc_diffs) if auc_diffs else np.nan
        auc_se = np.std(auc_diffs) / np.sqrt(len(auc_diffs)) if len(auc_diffs) > 0 else 0
        
        durations = [x for x in data['success_duration'] if not np.isnan(x)]
        duration_mean = np.mean(durations) if durations else np.nan
        duration_se = np.std(durations) / np.sqrt(len(durations)) if len(durations) > 0 else 0
        
        all_food_densities.append(food_density)
        all_eating_values.append(eating_mean)
        all_eating_errors.append(eating_se)
        all_auc_diffs.append(auc_mean)
        all_auc_errors.append(auc_se)
        all_success_durations.append(duration_mean)
        all_duration_errors.append(duration_se)
        
        # Store data for CSV export
        all_plot_data.append({
            'experiment_type': 'food_density',
            'parameter_value': food_density,
            'parameter_name': 'food_density',
            'parameter_units': 'count/mm²',
            'eating_events': eating_mean,
            'eating_events_se': eating_se,
            'auc_difference': auc_mean,
            'auc_difference_se': auc_se,
            'success_duration': duration_mean * (1/cfg.ENV_PARAMS["fps_sim"]),  # Convert to seconds
            'success_duration_se': duration_se * (1/cfg.ENV_PARAMS["fps_sim"]),
            'env_id': env_id
        })

    # Convert to numpy arrays
    all_food_densities = np.array(all_food_densities)
    all_eating_values = np.array(all_eating_values)
    all_eating_errors = np.array(all_eating_errors)
    all_auc_diffs = np.array(all_auc_diffs)
    all_auc_errors = np.array(all_auc_errors)
    all_success_durations = np.array(all_success_durations)
    all_duration_errors = np.array(all_duration_errors)
    
    # Create plots directory
    plots_dir = os.path.join(outputs_folder, "additional_exps", "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Create three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 8))
    
    ax1.scatter(all_food_densities, all_eating_values, 
                alpha=0.6, s=50, c='blue', label='Food Density Experiments')
    ax1.errorbar(all_food_densities, all_eating_values, yerr=all_eating_errors, 
                 fmt='o', alpha=0.6, capsize=3, capthick=1, color='blue')
    # Add regression line
    valid_eating = ~np.isnan(all_eating_values)
    if np.sum(valid_eating) > 1:
        z = np.polyfit(all_food_densities[valid_eating], all_eating_values[valid_eating], 1)
        p = np.poly1d(z)
        ax1.plot(all_food_densities[valid_eating], p(all_food_densities[valid_eating]), "b--", alpha=0.8)
    # Add control point for comparison
    if 'control' in dff_dict:
        control_food_density = dff_dict['control']['reset_food_density'].iloc[0]
        ax1.scatter(control_food_density, avg_eating_per_episode, 
                    alpha=0.8, s=100, c='red', marker='s', label='Control', edgecolors='black')
        ax1.errorbar(control_food_density, avg_eating_per_episode, yerr=std_eating_per_episode, 
                        fmt='s', markersize=10, color='red', alpha=0.8, capsize=5, capthick=2)
    ax1.set_xlabel('Food Density (count/mm²)', fontsize=16)
    ax1.set_ylabel('Eating Events per Episode', fontsize=16)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=14)
    ax1.tick_params(axis='both', which='major', labelsize=14)
    
    # Plot 2: AUC Difference vs Food Density
    valid_auc = ~np.isnan(all_auc_diffs)
    ax2.scatter(all_food_densities[valid_auc], all_auc_diffs[valid_auc], 
                alpha=0.6, s=50, c='blue', label='Food Density Experiments')
    ax2.errorbar(all_food_densities[valid_auc], all_auc_diffs[valid_auc], yerr=all_auc_errors[valid_auc], 
                 fmt='o', alpha=0.6, capsize=3, capthick=1, color='blue')
    # Add regression line
    if np.sum(valid_auc) > 1:
        z = np.polyfit(all_food_densities[valid_auc], all_auc_diffs[valid_auc], 1)
        p = np.poly1d(z)
        ax2.plot(all_food_densities[valid_auc], p(all_food_densities[valid_auc]), "b--", alpha=0.8)
    if 'control' in dff_dict:
        ax2.scatter(control_food_density, control_auc_difference, 
                    alpha=0.8, s=100, c='red', marker='s', label='Control', edgecolors='black')
        ax2.errorbar(control_food_density, control_auc_difference, yerr=control_auc_se, 
                        fmt='s', markersize=10, color='red', alpha=0.8, capsize=5, capthick=2)
    ax2.set_xlabel('Food Density (count/mm²)', fontsize=16)
    ax2.set_ylabel('AUC Difference (°)', fontsize=16)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=14)
    ax2.tick_params(axis='both', which='major', labelsize=14)
    
    # Plot 3: Success Duration vs Food Density
    valid_duration = ~np.isnan(all_success_durations)
    all_success_durations = all_success_durations / cfg.ENV_PARAMS["fps_sim"]  # Convert to seconds
    ax3.scatter(all_food_densities[valid_duration], all_success_durations[valid_duration], 
                alpha=0.6, s=50, c='blue', label='Food Density Experiments')
    ax3.errorbar(all_food_densities[valid_duration], all_success_durations[valid_duration], yerr=all_duration_errors[valid_duration], 
                 fmt='o', alpha=0.6, capsize=3, capthick=1, color='blue')
    # Add regression line
    if np.sum(valid_duration) > 1:
        z = np.polyfit(all_food_densities[valid_duration], all_success_durations[valid_duration], 1)
        p = np.poly1d(z)
        ax3.plot(all_food_densities[valid_duration], p(all_food_densities[valid_duration]), "b--", alpha=0.8)
    if 'control' in dff_dict:
        ax3.scatter(control_food_density, avg_success_duration, 
                    alpha=0.8, s=100, c='red', marker='s', label='Control', edgecolors='black')
        ax3.errorbar(control_food_density, avg_success_duration, yerr=std_success_duration, 
                        fmt='s', markersize=10, color='red', alpha=0.8, capsize=5, capthick=2)
    ax3.set_xlabel('Food Density (count/mm²)', fontsize=16)
    ax3.set_ylabel('Average Success Duration (s)', fontsize=16)
    ax3.set_title('Success Duration vs Food Density', fontsize=18)
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=14)
    ax3.tick_params(axis='both', which='major', labelsize=14)
    
    plt.tight_layout()
    
    # Save the plot
    plot_path = os.path.join(plots_dir, "episode_level_food_density_analysis.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\nEpisode-level analysis plot saved to: {plot_path}")
    plt.show()
        
else:
    print("\nNo dataframes found with food_density in name")

if 'limit_convergence' in dff_dict:
    limit_convergence_df = dff_dict['limit_convergence']
    print(f"\nAnalyzing limit_convergence dataframe.")
    print(f"Shape: {limit_convergence_df.shape}")

    # Dictionary to store data by limit convergence and env_id
    env_data = {}
    
    # Get unique episodes
    episodes = limit_convergence_df.groupby(['env_id', 'episode_index'])
    
    for (env_id, episode_index), episode_df in episodes:
        # Get limit convergence for this episode (should be constant within episode)
        limit_convergence = np.abs((episode_df['max_left_vergence'].iloc[0] - cfg.FISH_CONSTANTS["max_left_vergence"]))
        # Calculate eating events for this episode
        eating_events = episode_df['eating_event'].sum()
        
        # Calculate vergence analysis for this episode
        tracking_sequences_df = analyze_vergence_during_food_tracking(episode_df)
        
        if len(tracking_sequences_df) > 0:
            non_tracking_vergence = non_tracking_data(episode_df, tracking_sequences_df)
            success_trajectories, miss_trajectories = calculate_avg_vergence_by_outcome(tracking_sequences_df)
            
            if len(success_trajectories) > 0 and len(non_tracking_vergence) > 0:
                avg_auc, non_tracking_auc = calculate_auc_hunting_no_hunting(success_trajectories, non_tracking_vergence)                
                # Convert AUC to degrees
                max_vergence = (cfg.FISH_CONSTANTS["max_left_vergence"] - cfg.FISH_CONSTANTS["max_right_vergence"] + cfg.FISH_CONSTANTS["perception_field"]) * 180 / np.pi
                avg_auc_degrees = avg_auc * max_vergence
                non_tracking_auc_degrees = non_tracking_auc * max_vergence
                auc_difference = avg_auc_degrees - non_tracking_auc_degrees
                
                # Calculate successful hunting duration
                success_data = tracking_sequences_df[tracking_sequences_df['outcome'] == 'success']
                if len(success_data) > 0:
                    success_duration = success_data['tracking_duration'].mean()
                else:
                    success_duration = np.nan
                
                # Store data by env_id and limit_convergence
                key = (env_id, limit_convergence)
                if key not in env_data:
                    env_data[key] = {'eating': [], 'auc_diff': [], 'success_duration': []}
                
                env_data[key]['eating'].append(eating_events)
                env_data[key]['auc_diff'].append(auc_difference)
                env_data[key]['success_duration'].append(success_duration)

    # Calculate averages and standard errors for each environment
    all_limit_convergences = []
    all_eating_values = []
    all_eating_errors = []
    all_auc_diffs = []
    all_auc_errors = []
    all_success_durations = []
    all_duration_errors = []
    
    for (env_id, limit_convergence), data in env_data.items():
        # Calculate means and standard errors
        eating_mean = np.mean(data['eating'])
        eating_se = np.std(data['eating']) / np.sqrt(len(data['eating']))
        
        auc_diffs = [x for x in data['auc_diff'] if not np.isnan(x)]
        auc_mean = np.mean(auc_diffs) if auc_diffs else np.nan
        auc_se = np.std(auc_diffs) / np.sqrt(len(auc_diffs)) if len(auc_diffs) > 0 else 0
        
        durations = [x for x in data['success_duration'] if not np.isnan(x)]
        duration_mean = np.mean(durations) if durations else np.nan
        duration_se = np.std(durations) / np.sqrt(len(durations)) if len(durations) > 0 else 0
        
        all_limit_convergences.append(limit_convergence)
        all_eating_values.append(eating_mean)
        all_eating_errors.append(eating_se)
        all_auc_diffs.append(auc_mean)
        all_auc_errors.append(auc_se)
        all_success_durations.append(duration_mean)
        all_duration_errors.append(duration_se)
        
        # Store data for CSV export
        all_plot_data.append({
            'experiment_type': 'limit_convergence',
            'parameter_value': limit_convergence * (180 / np.pi),  # Convert to degrees
            'parameter_name': 'delta_max_vergence',
            'parameter_units': 'degrees',
            'eating_events': eating_mean,
            'eating_events_se': eating_se,
            'auc_difference': auc_mean,
            'auc_difference_se': auc_se,
            'success_duration': duration_mean * (1/cfg.ENV_PARAMS["fps_sim"]),  # Convert to seconds
            'success_duration_se': duration_se * (1/cfg.ENV_PARAMS["fps_sim"]),
            'env_id': env_id
        })

    # Convert to numpy arrays
    all_limit_convergences = np.array(all_limit_convergences)
    all_eating_values = np.array(all_eating_values)
    all_eating_errors = np.array(all_eating_errors)
    all_auc_diffs = np.array(all_auc_diffs)
    all_auc_errors = np.array(all_auc_errors)
    all_success_durations = np.array(all_success_durations)
    all_duration_errors = np.array(all_duration_errors)
    
    # Create plots directory
    plots_dir = os.path.join(outputs_folder, "additional_exps", "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Create three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 8))

    all_limit_convergences = all_limit_convergences * (180 / np.pi)  # Convert to degrees for plotting
    ax1.scatter(all_limit_convergences, all_eating_values, 
                alpha=0.6, s=50, c='blue', label='Limit Convergence Experiments')
    ax1.errorbar(all_limit_convergences, all_eating_values, yerr=all_eating_errors, 
                 fmt='o', alpha=0.6, capsize=3, capthick=1, color='blue')
    # Add regression line
    valid_eating = ~np.isnan(all_eating_values)
    if np.sum(valid_eating) > 1:
        z = np.polyfit(all_limit_convergences[valid_eating], all_eating_values[valid_eating], 1)
        p = np.poly1d(z)
        ax1.plot(all_limit_convergences[valid_eating], p(all_limit_convergences[valid_eating]), "b--", alpha=0.8)
    # Add control point for comparison
    control_limit_convergence = np.abs((dff_dict['control']['max_left_vergence'].iloc[0] - cfg.FISH_CONSTANTS["max_left_vergence"]))
    if 'control' in dff_dict:
        ax1.scatter(control_limit_convergence, avg_eating_per_episode, 
                    alpha=0.8, s=100, c='red', marker='s', label='Control', edgecolors='black')
        ax1.errorbar(control_limit_convergence, avg_eating_per_episode, yerr=std_eating_per_episode, 
                        fmt='s', markersize=10, color='red', alpha=0.8, capsize=5, capthick=2)
    ax1.set_xlabel('Δ Max Vergence (°)', fontsize=16)
    ax1.set_ylabel('Eating Events per Episode', fontsize=16)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=14)
    ax1.tick_params(axis='both', which='major', labelsize=14)
    
    # Plot 2: AUC Difference vs Limit Convergence
    valid_auc = ~np.isnan(all_auc_diffs)
    ax2.scatter(all_limit_convergences[valid_auc], all_auc_diffs[valid_auc], 
                alpha=0.6, s=50, c='blue', label='Limit Convergence Experiments')
    ax2.errorbar(all_limit_convergences[valid_auc], all_auc_diffs[valid_auc], yerr=all_auc_errors[valid_auc], 
                 fmt='o', alpha=0.6, capsize=3, capthick=1, color='blue')
    # Add regression line
    if np.sum(valid_auc) > 1:
        z = np.polyfit(all_limit_convergences[valid_auc], all_auc_diffs[valid_auc], 1)
        p = np.poly1d(z)
        ax2.plot(all_limit_convergences[valid_auc], p(all_limit_convergences[valid_auc]), "b--", alpha=0.8)
    if 'control' in dff_dict:
        ax2.scatter(control_limit_convergence, control_auc_difference, 
                    alpha=0.8, s=100, c='red', marker='s', label='Control', edgecolors='black')
        ax2.errorbar(control_limit_convergence, control_auc_difference, yerr=control_auc_se, 
                        fmt='s', markersize=10, color='red', alpha=0.8, capsize=5, capthick=2)
    ax1.set_xlabel('Δ Max Vergence (°)', fontsize=16)
    ax2.set_ylabel('AUC Difference (°)', fontsize=16)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=14)
    ax2.tick_params(axis='both', which='major', labelsize=14)
    
    # Plot 3: Success Duration vs Limit Convergence
    valid_duration = ~np.isnan(all_success_durations)
    all_success_durations = all_success_durations / cfg.ENV_PARAMS["fps_sim"]  # Convert to seconds
    ax3.scatter(all_limit_convergences[valid_duration], all_success_durations[valid_duration], 
                alpha=0.6, s=50, c='blue', label='Limit Convergence Experiments')
    ax3.errorbar(all_limit_convergences[valid_duration], all_success_durations[valid_duration], yerr=all_duration_errors[valid_duration], 
                 fmt='o', alpha=0.6, capsize=3, capthick=1, color='blue')
    # Add regression line
    if np.sum(valid_duration) > 1:
        z = np.polyfit(all_limit_convergences[valid_duration], all_success_durations[valid_duration], 1)
        p = np.poly1d(z)
        ax3.plot(all_limit_convergences[valid_duration], p(all_limit_convergences[valid_duration]), "b--", alpha=0.8)
    if 'control' in dff_dict:
        ax3.scatter(control_limit_convergence, avg_success_duration, 
                    alpha=0.8, s=100, c='red', marker='s', label='Control', edgecolors='black')
        ax3.errorbar(control_limit_convergence, avg_success_duration, yerr=std_success_duration, 
                        fmt='s', markersize=10, color='red', alpha=0.8, capsize=5, capthick=2)
    ax1.set_xlabel('Δ Max Vergence (°)', fontsize=16)
    ax3.set_ylabel('Average Success Duration (s)', fontsize=16)
    ax3.set_title('Success Duration vs Δ Max Vergence', fontsize=18)
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=14)
    ax3.tick_params(axis='both', which='major', labelsize=14)
    
    plt.tight_layout()
    
    # Save the plot
    plot_path = os.path.join(plots_dir, "episode_level_limit_convergence_analysis.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\nEpisode-level analysis plot saved to: {plot_path}")
    plt.show()
        
else:
    print("\nNo dataframes found with limit_convergence in name")

if 'limit_divergence' in dff_dict:
    limit_divergence_df = dff_dict['limit_divergence']
    print(f"\nAnalyzing limit_divergence dataframe.")
    print(f"Shape: {limit_divergence_df.shape}")

    # Dictionary to store data by limit divergence and env_id
    env_data = {}
    
    # Get unique episodes
    episodes = limit_divergence_df.groupby(['env_id', 'episode_index'])
    
    for (env_id, episode_index), episode_df in episodes:
        # Get limit divergence for this episode (should be constant within episode)
        limit_divergence = np.abs((episode_df['min_left_vergence'].iloc[0] - cfg.FISH_CONSTANTS["min_left_vergence"]))

        # Calculate eating events for this episode
        eating_events = episode_df['eating_event'].sum()
        
        # Calculate vergence analysis for this episode
        tracking_sequences_df = analyze_vergence_during_food_tracking(episode_df)
        
        if len(tracking_sequences_df) > 0:
            non_tracking_vergence = non_tracking_data(episode_df, tracking_sequences_df)
            success_trajectories, miss_trajectories = calculate_avg_vergence_by_outcome(tracking_sequences_df)
            
            if len(success_trajectories) > 0 and len(non_tracking_vergence) > 0:
                avg_auc, non_tracking_auc = calculate_auc_hunting_no_hunting(success_trajectories, non_tracking_vergence)
                
                # Convert AUC to degrees
                max_vergence = (cfg.FISH_CONSTANTS["max_left_vergence"] - cfg.FISH_CONSTANTS["max_right_vergence"] + cfg.FISH_CONSTANTS["perception_field"]) * 180 / np.pi
                avg_auc_degrees = avg_auc * max_vergence
                non_tracking_auc_degrees = non_tracking_auc * max_vergence
                auc_difference = avg_auc_degrees - non_tracking_auc_degrees
                
                # Calculate successful hunting duration
                success_data = tracking_sequences_df[tracking_sequences_df['outcome'] == 'success']
                if len(success_data) > 0:
                    success_duration = success_data['tracking_duration'].mean()
                else:
                    success_duration = np.nan
                
                # Store data by env_id and limit_divergence
                key = (env_id, limit_divergence)
                if key not in env_data:
                    env_data[key] = {'eating': [], 'auc_diff': [], 'success_duration': []}
                
                env_data[key]['eating'].append(eating_events)
                env_data[key]['auc_diff'].append(auc_difference)
                env_data[key]['success_duration'].append(success_duration)

    # Calculate averages and standard errors for each environment
    all_limit_divergences = []
    all_eating_values = []
    all_eating_errors = []
    all_auc_diffs = []
    all_auc_errors = []
    all_success_durations = []
    all_duration_errors = []
    
    for (env_id, limit_divergence), data in env_data.items():
        # Calculate means and standard errors
        eating_mean = np.mean(data['eating'])
        eating_se = np.std(data['eating']) / np.sqrt(len(data['eating']))
        
        auc_diffs = [x for x in data['auc_diff'] if not np.isnan(x)]
        auc_mean = np.mean(auc_diffs) if auc_diffs else np.nan
        auc_se = np.std(auc_diffs) / np.sqrt(len(auc_diffs)) if len(auc_diffs) > 0 else 0
        
        durations = [x for x in data['success_duration'] if not np.isnan(x)]
        duration_mean = np.mean(durations) if durations else np.nan
        duration_se = np.std(durations) / np.sqrt(len(durations)) if len(durations) > 0 else 0
        
        all_limit_divergences.append(limit_divergence)
        all_eating_values.append(eating_mean)
        all_eating_errors.append(eating_se)
        all_auc_diffs.append(auc_mean)
        all_auc_errors.append(auc_se)
        all_success_durations.append(duration_mean)
        all_duration_errors.append(duration_se)
        
        # Store data for CSV export
        all_plot_data.append({
            'experiment_type': 'limit_divergence',
            'parameter_value': limit_divergence * (180 / np.pi),  # Convert to degrees
            'parameter_name': 'delta_min_vergence',
            'parameter_units': 'degrees',
            'eating_events': eating_mean,
            'eating_events_se': eating_se,
            'auc_difference': auc_mean,
            'auc_difference_se': auc_se,
            'success_duration': duration_mean * (1/cfg.ENV_PARAMS["fps_sim"]),  # Convert to seconds
            'success_duration_se': duration_se * (1/cfg.ENV_PARAMS["fps_sim"]),
            'env_id': env_id
        })

    # Convert to numpy arrays
    all_limit_divergences = np.array(all_limit_divergences)
    all_eating_values = np.array(all_eating_values)
    all_eating_errors = np.array(all_eating_errors)
    all_auc_diffs = np.array(all_auc_diffs)
    all_auc_errors = np.array(all_auc_errors)
    all_success_durations = np.array(all_success_durations)
    all_duration_errors = np.array(all_duration_errors)
    
    # Create plots directory
    plots_dir = os.path.join(outputs_folder, "additional_exps", "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Create three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 8))
    
    all_limit_divergences = all_limit_divergences * (180 / np.pi)  # Convert to degrees for plotting
    ax1.scatter(all_limit_divergences, all_eating_values, 
                alpha=0.6, s=50, c='blue', label='Limit Divergence Experiments')
    ax1.errorbar(all_limit_divergences, all_eating_values, yerr=all_eating_errors, 
                 fmt='o', alpha=0.6, capsize=3, capthick=1, color='blue')
    # Add regression line
    valid_eating = ~np.isnan(all_eating_values)
    if np.sum(valid_eating) > 1:
        z = np.polyfit(all_limit_divergences[valid_eating], all_eating_values[valid_eating], 1)
        p = np.poly1d(z)
        ax1.plot(all_limit_divergences[valid_eating], p(all_limit_divergences[valid_eating]), "b--", alpha=0.8)
    # Add control point for comparison
    control_limit_divergence = np.abs((dff_dict['control']['min_left_vergence'].iloc[0] - cfg.FISH_CONSTANTS["min_left_vergence"]))
    if 'control' in dff_dict:
        ax1.scatter(control_limit_divergence, avg_eating_per_episode, 
                    alpha=0.8, s=100, c='red', marker='s', label='Control', edgecolors='black')
        ax1.errorbar(control_limit_divergence, avg_eating_per_episode, yerr=std_eating_per_episode, 
                        fmt='s', markersize=10, color='red', alpha=0.8, capsize=5, capthick=2)
    ax1.set_xlabel('Δ Min Vergence (°)', fontsize=16)
    ax1.set_ylabel('Eating Events per Episode', fontsize=16)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=14)
    ax1.tick_params(axis='both', which='major', labelsize=14)
    
    # Plot 2: AUC Difference vs Limit Divergence
    valid_auc = ~np.isnan(all_auc_diffs)
    ax2.scatter(all_limit_divergences[valid_auc], all_auc_diffs[valid_auc], 
                alpha=0.6, s=50, c='blue', label='Limit Divergence Experiments')
    ax2.errorbar(all_limit_divergences[valid_auc], all_auc_diffs[valid_auc], yerr=all_auc_errors[valid_auc], 
                 fmt='o', alpha=0.6, capsize=3, capthick=1, color='blue')
    # Add regression line
    if np.sum(valid_auc) > 1:
        z = np.polyfit(all_limit_divergences[valid_auc], all_auc_diffs[valid_auc], 1)
        p = np.poly1d(z)
        ax2.plot(all_limit_divergences[valid_auc], p(all_limit_divergences[valid_auc]), "b--", alpha=0.8)
    if 'control' in dff_dict:
        ax2.scatter(control_limit_divergence, control_auc_difference, 
                    alpha=0.8, s=100, c='red', marker='s', label='Control', edgecolors='black')
        ax2.errorbar(control_limit_divergence, control_auc_difference, yerr=control_auc_se, 
                        fmt='s', markersize=10, color='red', alpha=0.8, capsize=5, capthick=2)
    ax1.set_xlabel('Δ Min Vergence (°)', fontsize=16)
    ax2.set_ylabel('AUC Difference (°)', fontsize=16)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=14)
    ax2.tick_params(axis='both', which='major', labelsize=14)
    
    # Plot 3: Success Duration vs Limit Divergence
    valid_duration = ~np.isnan(all_success_durations)
    all_success_durations = all_success_durations / cfg.ENV_PARAMS["fps_sim"]  # Convert to seconds
    ax3.scatter(all_limit_divergences[valid_duration], all_success_durations[valid_duration], 
                alpha=0.6, s=50, c='blue', label='Limit Divergence Experiments')
    ax3.errorbar(all_limit_divergences[valid_duration], all_success_durations[valid_duration], yerr=all_duration_errors[valid_duration], 
                 fmt='o', alpha=0.6, capsize=3, capthick=1, color='blue')
    # Add regression line
    if np.sum(valid_duration) > 1:
        z = np.polyfit(all_limit_divergences[valid_duration], all_success_durations[valid_duration], 1)
        p = np.poly1d(z)
        ax3.plot(all_limit_divergences[valid_duration], p(all_limit_divergences[valid_duration]), "b--", alpha=0.8)
    if 'control' in dff_dict:
        ax3.scatter(control_limit_divergence, avg_success_duration, 
                    alpha=0.8, s=100, c='red', marker='s', label='Control', edgecolors='black')
        ax3.errorbar(control_limit_divergence, avg_success_duration, yerr=std_success_duration, 
                        fmt='s', markersize=10, color='red', alpha=0.8, capsize=5, capthick=2)
    ax1.set_xlabel('Δ Min Vergence (°)', fontsize=16)
    ax3.set_ylabel('Average Success Duration (s)', fontsize=16)
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=14)
    ax3.tick_params(axis='both', which='major', labelsize=14)
    
    plt.tight_layout()
    
    # Save the plot
    plot_path = os.path.join(plots_dir, "episode_level_limit_divergence_analysis.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\nEpisode-level analysis plot saved to: {plot_path}")
    plt.show()
        
else:
    print("\nNo dataframes found with limit_divergence in name")

if 'num_bots' in dff_dict:
    num_bots_df = dff_dict['num_bots']
    print(f"\nAnalyzing num_bots dataframe.")
    print(f"Shape: {num_bots_df.shape}")

    # Dictionary to store data by num_bots and env_id
    env_data = {}
    
    # Get unique episodes
    episodes = num_bots_df.groupby(['env_id', 'episode_index'])
    
    for (env_id, episode_index), episode_df in episodes:
        # Get num_bots for this episode (should be constant within episode)
        num_bots = episode_df['num_bots'].iloc[0] if 'num_bots' in episode_df.columns else np.nan

        # Calculate eating events for this episode
        eating_events = episode_df['eating_event'].sum()
        
        # Calculate vergence analysis for this episode
        tracking_sequences_df = analyze_vergence_during_food_tracking(episode_df)
        
        if len(tracking_sequences_df) > 0:
            non_tracking_vergence = non_tracking_data(episode_df, tracking_sequences_df)
            success_trajectories, miss_trajectories = calculate_avg_vergence_by_outcome(tracking_sequences_df)
            
            if len(success_trajectories) > 0 and len(non_tracking_vergence) > 0:
                avg_auc, non_tracking_auc = calculate_auc_hunting_no_hunting(success_trajectories, non_tracking_vergence)
                
                # Convert AUC to degrees
                max_vergence = (cfg.FISH_CONSTANTS["max_left_vergence"] - cfg.FISH_CONSTANTS["max_right_vergence"] + cfg.FISH_CONSTANTS["perception_field"]) * 180 / np.pi
                avg_auc_degrees = avg_auc * max_vergence
                non_tracking_auc_degrees = non_tracking_auc * max_vergence
                auc_difference = avg_auc_degrees - non_tracking_auc_degrees
                
                # Calculate successful hunting duration
                success_data = tracking_sequences_df[tracking_sequences_df['outcome'] == 'success']
                if len(success_data) > 0:
                    success_duration = success_data['tracking_duration'].mean()
                else:
                    success_duration = np.nan
                
                # Store data by env_id and num_bots
                key = (env_id, num_bots)
                if key not in env_data:
                    env_data[key] = {'eating': [], 'auc_diff': [], 'success_duration': []}
                
                env_data[key]['eating'].append(eating_events)
                env_data[key]['auc_diff'].append(auc_difference)
                env_data[key]['success_duration'].append(success_duration)

    # Calculate averages and standard errors for each environment
    all_num_bots = []
    all_eating_values = []
    all_eating_errors = []
    all_auc_diffs = []
    all_auc_errors = []
    all_success_durations = []
    all_duration_errors = []
    
    for (env_id, num_bots), data in env_data.items():
        # Calculate means and standard errors
        eating_mean = np.mean(data['eating'])
        eating_se = np.std(data['eating']) / np.sqrt(len(data['eating']))
        
        auc_diffs = [x for x in data['auc_diff'] if not np.isnan(x)]
        auc_mean = np.mean(auc_diffs) if auc_diffs else np.nan
        auc_se = np.std(auc_diffs) / np.sqrt(len(auc_diffs)) if len(auc_diffs) > 0 else 0
        
        durations = [x for x in data['success_duration'] if not np.isnan(x)]
        duration_mean = np.mean(durations) if durations else np.nan
        duration_se = np.std(durations) / np.sqrt(len(durations)) if len(durations) > 0 else 0
        
        all_num_bots.append(num_bots)
        all_eating_values.append(eating_mean)
        all_eating_errors.append(eating_se)
        all_auc_diffs.append(auc_mean)
        all_auc_errors.append(auc_se)
        all_success_durations.append(duration_mean)
        all_duration_errors.append(duration_se)
        
        # Store data for CSV export
        all_plot_data.append({
            'experiment_type': 'num_bots',
            'parameter_value': num_bots,
            'parameter_name': 'num_bots',
            'parameter_units': 'count',
            'eating_events': eating_mean,
            'eating_events_se': eating_se,
            'auc_difference': auc_mean,
            'auc_difference_se': auc_se,
            'success_duration': duration_mean * (1/cfg.ENV_PARAMS["fps_sim"]),  # Convert to seconds
            'success_duration_se': duration_se * (1/cfg.ENV_PARAMS["fps_sim"]),
            'env_id': env_id
        })

    # Convert to numpy arrays
    all_num_bots = np.array(all_num_bots)
    all_eating_values = np.array(all_eating_values)
    all_eating_errors = np.array(all_eating_errors)
    all_auc_diffs = np.array(all_auc_diffs)
    all_auc_errors = np.array(all_auc_errors)
    all_success_durations = np.array(all_success_durations)
    all_duration_errors = np.array(all_duration_errors)
    
    # Create plots directory
    plots_dir = os.path.join(outputs_folder, "additional_exps", "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Create three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 8))
    
    ax1.scatter(all_num_bots, all_eating_values, 
                alpha=0.6, s=50, c='blue', label='Num Bots Experiments')
    ax1.errorbar(all_num_bots, all_eating_values, yerr=all_eating_errors, 
                    fmt='o', alpha=0.6, capsize=3, capthick=1, color='blue')
    # Add regression line
    valid_eating = ~np.isnan(all_eating_values)
    if np.sum(valid_eating) > 1:
        z = np.polyfit(all_num_bots[valid_eating], all_eating_values[valid_eating], 1)
        p = np.poly1d(z)
        ax1.plot(all_num_bots[valid_eating], p(all_num_bots[valid_eating]), "b--", alpha=0.8)
    # Add control point for comparison
    control_num_bots = cfg.ENV_PARAMS["num_bots"] if 'num_bots' in cfg.ENV_PARAMS else 1
    if 'control' in dff_dict:
        ax1.scatter(control_num_bots, avg_eating_per_episode, 
                    alpha=0.8, s=100, c='red', marker='s', label='Control', edgecolors='black')
        ax1.errorbar(control_num_bots, avg_eating_per_episode, yerr=std_eating_per_episode, 
                        fmt='s', markersize=10, color='red', alpha=0.8, capsize=5, capthick=2)
    ax1.set_xlabel('Number of Bots', fontsize=16)
    ax1.set_ylabel('Eating Events per Episode', fontsize=16)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=14)
    ax1.tick_params(axis='both', which='major', labelsize=14)
    
    # Plot 2: AUC Difference vs Num Bots
    valid_auc = ~np.isnan(all_auc_diffs)
    ax2.scatter(all_num_bots[valid_auc], all_auc_diffs[valid_auc], 
                alpha=0.6, s=50, c='blue', label='Num Bots Experiments')
    ax2.errorbar(all_num_bots[valid_auc], all_auc_diffs[valid_auc], yerr=all_auc_errors[valid_auc], 
                    fmt='o', alpha=0.6, capsize=3, capthick=1, color='blue')
    # Add regression line
    if np.sum(valid_auc) > 1:
        z = np.polyfit(all_num_bots[valid_auc], all_auc_diffs[valid_auc], 1)
        p = np.poly1d(z)
        ax2.plot(all_num_bots[valid_auc], p(all_num_bots[valid_auc]), "b--", alpha=0.8)
    if 'control' in dff_dict:
        ax2.scatter(control_num_bots, control_auc_difference, 
                    alpha=0.8, s=100, c='red', marker='s', label='Control', edgecolors='black')
        ax2.errorbar(control_num_bots, control_auc_difference, yerr=control_auc_se, 
                        fmt='s', markersize=10, color='red', alpha=0.8, capsize=5, capthick=2)
    ax2.set_xlabel('Number of Bots', fontsize=16)
    ax2.set_ylabel('AUC Difference (°)', fontsize=16)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=14)
    ax2.tick_params(axis='both', which='major', labelsize=14)
    
    # Plot 3: Success Duration vs Num Bots
    valid_duration = ~np.isnan(all_success_durations)
    all_success_durations = all_success_durations / cfg.ENV_PARAMS["fps_sim"]  # Convert to seconds
    ax3.scatter(all_num_bots[valid_duration], all_success_durations[valid_duration], 
                alpha=0.6, s=50, c='blue', label='Num Bots Experiments')
    ax3.errorbar(all_num_bots[valid_duration], all_success_durations[valid_duration], yerr=all_duration_errors[valid_duration], 
                    fmt='o', alpha=0.6, capsize=3, capthick=1, color='blue')
    # Add regression line
    if np.sum(valid_duration) > 1:
        z = np.polyfit(all_num_bots[valid_duration], all_success_durations[valid_duration], 1)
        p = np.poly1d(z)
        ax3.plot(all_num_bots[valid_duration], p(all_num_bots[valid_duration]), "b--", alpha=0.8)
    if 'control' in dff_dict:
        ax3.scatter(control_num_bots, avg_success_duration, 
                    alpha=0.8, s=100, c='red', marker='s', label='Control', edgecolors='black')
        ax3.errorbar(control_num_bots, avg_success_duration, yerr=std_success_duration, 
                        fmt='s', markersize=10, color='red', alpha=0.8, capsize=5, capthick=2)
    ax3.set_xlabel('Number of Bots', fontsize=16)
    ax3.set_ylabel('Average Success Duration (s)', fontsize=16)
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=14)
    ax3.tick_params(axis='both', which='major', labelsize=14)
    
    plt.tight_layout()
    
    # Save the plot
    plot_path = os.path.join(plots_dir, "episode_level_num_bots_analysis.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\nEpisode-level analysis plot saved to: {plot_path}")
    plt.show()
        
else:
    print("\nNo dataframes found with num_bots in name")

# Add control data to plot data if available
if 'control' in dff_dict:
    control_food_speed = cfg.ENV_PARAMS["food_speed"]
    control_food_density = dff_dict['control']['reset_food_density'].iloc[0]
    control_limit_convergence = np.abs((dff_dict['control']['max_left_vergence'].iloc[0] - cfg.FISH_CONSTANTS["max_left_vergence"]))
    control_limit_divergence = np.abs((dff_dict['control']['min_left_vergence'].iloc[0] - cfg.FISH_CONSTANTS["min_left_vergence"]))
    
    # Add control entries for each experiment type
    all_plot_data.append({
        'experiment_type': 'control_food_speed',
        'parameter_value': control_food_speed * cfg.ENV_PARAMS["fps_sim"],  # Convert to mm/s
        'parameter_name': 'food_speed',
        'parameter_units': 'mm/s',
        'eating_events': avg_eating_per_episode,
        'eating_events_se': std_eating_per_episode,
        'auc_difference': control_auc_difference,
        'auc_difference_se': control_auc_se,
        'success_duration': avg_success_duration,
        'success_duration_se': std_success_duration,
        'env_id': 'control'
    })
    
    all_plot_data.append({
        'experiment_type': 'control_food_density',
        'parameter_value': control_food_density,
        'parameter_name': 'food_density',
        'parameter_units': 'count/mm²',
        'eating_events': avg_eating_per_episode,
        'eating_events_se': std_eating_per_episode,
        'auc_difference': control_auc_difference,
        'auc_difference_se': control_auc_se,
        'success_duration': avg_success_duration,
        'success_duration_se': std_success_duration,
        'env_id': 'control'
    })
    
    all_plot_data.append({
        'experiment_type': 'control_limit_convergence',
        'parameter_value': control_limit_convergence * (180 / np.pi),
        'parameter_name': 'delta_max_vergence',
        'parameter_units': 'degrees',
        'eating_events': avg_eating_per_episode,
        'eating_events_se': std_eating_per_episode,
        'auc_difference': control_auc_difference,
        'auc_difference_se': control_auc_se,
        'success_duration': avg_success_duration,
        'success_duration_se': std_success_duration,
        'env_id': 'control'
    })
    
    all_plot_data.append({
        'experiment_type': 'control_limit_divergence',
        'parameter_value': control_limit_divergence * (180 / np.pi),
        'parameter_name': 'delta_min_vergence',
        'parameter_units': 'degrees',
        'eating_events': avg_eating_per_episode,
        'eating_events_se': std_eating_per_episode,
        'auc_difference': control_auc_difference,
        'auc_difference_se': control_auc_se,
        'success_duration': avg_success_duration,
        'success_duration_se': std_success_duration,
        'env_id': 'control'
    })

# Save all plot data to CSV and text files
if all_plot_data:
    # Create additional_exps directory if it doesn't exist
    additional_exps_dir = os.path.join(outputs_folder, "additional_exps")
    os.makedirs(additional_exps_dir, exist_ok=True)
    
    # Convert to DataFrame and save as CSV
    plot_df = pd.DataFrame(all_plot_data)
    csv_path = os.path.join(additional_exps_dir, "all_experiment_data.csv")
    plot_df.to_csv(csv_path, index=False)
    print(f"\nAll experiment data saved to CSV: {csv_path}")
    print(f"Total data points saved: {len(all_plot_data)}")
else:
    print("No plot data to save.")

# # Filter dataframes that have food_speed in their key name
# food_speed_dfs = {name: df for name, df in dff_dict.items() if 'food_speed' in name}

# if food_speed_dfs:
#     print(f"\nFound {len(food_speed_dfs)} dataframes with food_speed in name")
    
#     # Calculate average eating events per episode for each dataframe
#     eating_stats = {}
#     for name, df in food_speed_dfs.items():
#         print(df.keys())
#         # Group by episode and count eating events per episode
#         eating_per_episode = df.groupby(['env_id', 'episode_index'])['eating_event'].sum()
#         avg_eating_per_episode = eating_per_episode.mean()
#         std_eating_per_episode = eating_per_episode.std() / (np.sqrt(len(eating_per_episode)))

#         print(f"{name}: {avg_eating_per_episode:.2f} ± {std_eating_per_episode:.2f} eating events per episode")

#         # Get cumulative reward at final time step for each episode
#         final_rewards_per_episode = df.groupby(['env_id', 'episode_index'])['cumulative_reward'].last()
#         avg_final_reward = final_rewards_per_episode.mean()
#         std_final_reward = final_rewards_per_episode.std() / (np.sqrt(len(final_rewards_per_episode)))
#         print(f"{name}: {avg_final_reward:.2f} ± {std_final_reward:.2f} average final cumulative reward per episode")

#         eating_stats[name] = (avg_eating_per_episode, std_eating_per_episode, avg_final_reward, std_final_reward)
    
#     # Extract food speed values from key names and create scatter plot
#     plt.figure(figsize=(10, 6))
    
#     food_speeds = []
#     eating_values = []
#     eating_std = []
#     reward_values = []

#     for name, (avg_eating, std_eating, avg_reward, std_reward) in eating_stats.items():
#         # Extract the numeric value after "food_speed_"
#         if 'food_speed_' in name:
#             try:
#                 # Find the part after "food_speed_" and extract the numeric value
#                 food_speed_str = name.split('food_speed_')[1]
#                 # Handle cases where there might be more text after the number
#                 food_speed_value = float(food_speed_str.split('_')[0])
#                 food_speeds.append(food_speed_value)
#                 eating_values.append(avg_eating)
#                 eating_std.append(std_eating)
#                 reward_values.append(avg_reward)
#             except (ValueError, IndexError):
#                 print(f"Could not extract food speed from: {name}")
    
#     if food_speeds:
#         plt.scatter(food_speeds, eating_values, s=100, alpha=0.7, label='Eating Events')
#         plt.errorbar(food_speeds, eating_values, yerr=eating_std, fmt='o', markersize=8, alpha=0.7, capsize=5, capthick=2)
#         # plt.scatter(food_speeds, reward_values, s=100, alpha=0.7, label='Cumulative Reward')
#         plt.title('Average Number of Eating Events per Episode vs Food Speed')
#         plt.xlabel('Food Speed')
#         plt.ylabel('Average Eating Events per Episode')
#         plt.grid(True, alpha=0.3)
#         plt.tight_layout()

#         # Add vertical line at food speed 1
#         plt.axvline(x=1, color='red', linestyle='--', alpha=0.8, linewidth=2)
#         plt.text(1.05, max(eating_values) * 0.9, 'Training food speed', 
#                  rotation=90, verticalalignment='top', fontsize=10, color='red')
        
#         # Create plots directory and save
#         plots_dir = os.path.join(outputs_folder, "additional_exps", "plots")
#         os.makedirs(plots_dir, exist_ok=True)
        
#         plot_path = os.path.join(plots_dir, "eating_events_vs_food_speed.png")
#         plt.savefig(plot_path, dpi=300, bbox_inches='tight')
#         print(f"\nScatter plot saved to: {plot_path}")
#     else:
#         print("Could not extract any food speed values from key names")
# else:
#     print("\nNo dataframes found with food_speed in name")

# # Filter dataframes that have food_density in their key name
# food_density_dfs = {name: df for name, df in dff_dict.items() if 'food_density' in name}
# # Add food_speed_1_control to food_density_dfs with renamed key
# if 'food_speed_1_control' in dff_dict:
#     food_density_dfs['food_density_0.003'] = dff_dict['food_speed_1_control']

# if food_density_dfs:
#     print(f"\nFound {len(food_density_dfs)} dataframes with food_density in name")
    
#     # Calculate average eating events per episode for each dataframe
#     eating_stats_density = {}
#     for name, df in food_density_dfs.items():
#         print(df.keys())
#         # Group by episode and count eating events per episode
#         eating_per_episode = df.groupby(['env_id', 'episode_index'])['eating_event'].sum()
#         avg_eating_per_episode = eating_per_episode.mean()
#         std_eating_per_episode = eating_per_episode.std() / (np.sqrt(len(eating_per_episode)))
#         print(f"{name}: {avg_eating_per_episode:.2f} ± {std_eating_per_episode:.2f} eating events per episode")

#         # Get cumulative reward at final time step for each episode
#         final_rewards_per_episode = df.groupby(['env_id', 'episode_index'])['cumulative_reward'].last()
#         avg_final_reward = final_rewards_per_episode.mean()
#         std_final_reward = final_rewards_per_episode.std() / (np.sqrt(len(final_rewards_per_episode)))
#         print(f"{name}: {avg_final_reward:.2f} ± {std_final_reward:.2f} average final cumulative reward per episode")


#         tracking_sequences_df = analyze_vergence_during_food_tracking(df)
#         non_tracking_vergence = non_tracking_data(df, tracking_sequences_df)
#         success_trajectories, miss_trajectories = calculate_avg_vergence_by_outcome(tracking_sequences_df)
#         avg_auc, non_tracking_auc = calculate_auc_hunting_no_hunting(success_trajectories, non_tracking_vergence)
                
#         success_data = tracking_sequences_df[tracking_sequences_df['outcome'] == 'success']
#         miss_data = tracking_sequences_df[tracking_sequences_df['outcome'] == 'miss']

#         # 1. Distribution of tracking durations by outcome (with log scale bins)
#         success_durations = success_data['tracking_duration'].values
#         avg_success_duration = np.mean(success_durations)
#         std_success_duration = np.std(success_durations) / (np.sqrt(len(success_durations)))
#         miss_durations = miss_data['tracking_duration'].values
#         # flatten success vergence angles
#         tracking_vergence = np.array([ang for _, angles in success_trajectories for ang in angles])

#         # use absolute vergence angles (to match your original)
#         trk = np.abs(tracking_vergence)
#         non = np.abs(np.asarray(non_tracking_vergence))

#         trk_std = np.std(trk)
#         non_std = np.std(non)

#         max_vergence = (cfg.FISH_CONSTANTS["max_left_vergence"] - cfg.FISH_CONSTANTS["max_right_vergence"] + cfg.FISH_CONSTANTS["perception_field"]) * 180 / np.pi
#         min_vergence = (cfg.FISH_CONSTANTS["min_left_vergence"] - cfg.FISH_CONSTANTS["min_right_vergence"] + cfg.FISH_CONSTANTS["perception_field"]) * 180 / np.pi

#         avg_auc = avg_auc * max_vergence
#         non_tracking_auc = non_tracking_auc * max_vergence

#         eating_stats_density[name] = (avg_eating_per_episode, std_eating_per_episode, avg_final_reward, std_final_reward, avg_auc, non_tracking_auc, trk_std, non_std, avg_success_duration, std_success_duration)

#     # Extract food density values from key names and create scatter plot
#     plt.figure(figsize=(10, 6))
    
#     food_densities = []
#     eating_values_density = []
#     eating_std_density = []
#     reward_values_density = []
#     auc_diffs = []
#     auc_diff_std = []
#     avg_success_durations = []
#     std_success_durations = []

#     for name, (avg_eating, std_eating, avg_reward, std_reward, avg_auc, non_tracking_auc, trk_std, non_std, avg_success_duration, std_success_duration) in eating_stats_density.items():
#         # Extract the numeric value after "food_density_"
#         if 'food_density_' in name:
#             try:
#                 # Find the part after "food_density_" and extract the numeric value
#                 food_density_str = name.split('food_density_')[1]
#                 # Handle cases where there might be more text after the number
#                 food_density_value = float(food_density_str.split('_')[0])
#                 food_densities.append(food_density_value)
#                 eating_values_density.append(avg_eating)
#                 eating_std_density.append(std_eating)
#                 auc_diffs.append(avg_auc - non_tracking_auc)
#                 auc_diff_std.append(np.sqrt(trk_std**2 + non_std**2))
#                 reward_values_density.append(avg_reward)
#                 avg_success_durations.append(avg_success_duration)
#                 std_success_durations.append(std_success_duration)
#             except (ValueError, IndexError):
#                 print(f"Could not extract food density from: {name}")
    
#     if food_densities:
#         # Plot 1: Eating Events
#         plt.figure(figsize=(10, 6))
#         plt.scatter(food_densities, eating_values_density, s=100, alpha=0.7, color='blue')
#         plt.errorbar(food_densities, eating_values_density, yerr=eating_std_density, fmt='o', markersize=8, alpha=0.7, capsize=5, capthick=2, color='blue')
#         plt.title('Average Number of Eating Events per Episode vs Food Density')
#         plt.xlabel('Food Density')
#         plt.ylabel('Average Eating Events per Episode')
#         plt.grid(True, alpha=0.3)
#         plt.axvline(0.003, color='red', linestyle='--', alpha=0.8, linewidth=2)
#         plt.text(0.0035, max(eating_values_density) * 0.9, 'Training food density', 
#                  rotation=90, verticalalignment='top', fontsize=10, color='red')
#         plt.tight_layout()
        
#         plots_dir = os.path.join(outputs_folder, "additional_exps", "plots")
#         os.makedirs(plots_dir, exist_ok=True)
#         plot_path = os.path.join(plots_dir, "eating_events_vs_food_density.png")
#         plt.savefig(plot_path, dpi=300, bbox_inches='tight')
#         print(f"\nEating events plot saved to: {plot_path}")
#         plt.show()

#         # Plot 2: AUC Differences
#         plt.figure(figsize=(10, 6))
#         plt.scatter(food_densities, auc_diffs, s=100, alpha=0.7, color='green')
#         plt.errorbar(food_densities, auc_diffs, yerr=auc_diff_std, fmt='o', markersize=8, alpha=0.7, capsize=5, capthick=2, color='green')
#         plt.title('AUC Differences vs Food Density')
#         plt.xlabel('Food Density')
#         plt.ylabel('AUC Difference (Tracking - Non-tracking)')
#         plt.grid(True, alpha=0.3)
#         plt.axvline(0.003, color='red', linestyle='--', alpha=0.8, linewidth=2)
#         plt.text(0.0035, max(auc_diffs) * 0.9, 'Training food density', 
#                  rotation=90, verticalalignment='top', fontsize=10, color='red')
#         plt.tight_layout()
        
#         plot_path = os.path.join(plots_dir, "auc_differences_vs_food_density.png")
#         plt.savefig(plot_path, dpi=300, bbox_inches='tight')
#         print(f"AUC differences plot saved to: {plot_path}")
#         plt.show()

#         # Plot 3: Average Success Duration
#         plt.figure(figsize=(10, 6))
#         plt.scatter(food_densities, avg_success_durations, s=100, alpha=0.7, color='orange')
#         plt.errorbar(food_densities, avg_success_durations, yerr=std_success_durations, fmt='o', markersize=8, alpha=0.7, capsize=5, capthick=2, color='orange')
#         plt.title('Average Success Duration vs Food Density')
#         plt.xlabel('Food Density')
#         plt.ylabel('Average Success Duration')
#         plt.grid(True, alpha=0.3)
#         plt.axvline(0.003, color='red', linestyle='--', alpha=0.8, linewidth=2)
#         plt.text(0.0035, max(avg_success_durations) * 0.9, 'Training food density', 
#                  rotation=90, verticalalignment='top', fontsize=10, color='red')
#         plt.tight_layout()
        
#         plot_path = os.path.join(plots_dir, "success_duration_vs_food_density.png")
#         plt.savefig(plot_path, dpi=300, bbox_inches='tight')
#         print(f"Success duration plot saved to: {plot_path}")
#         plt.show()
#     else:
#         print("Could not extract any food density values from key names")
# else:
#     print("\nNo dataframes found with food_density in name")

# # Filter dataframes that have vergence in their key name
# vergence_dfs = {name: df for name, df in dff_dict.items() if 'vergence' in name}
# # Add food_speed_1_control to vergence_dfs with renamed key
# if 'food_speed_1_control' in dff_dict:
#     vergence_dfs['vergence_control'] = dff_dict['food_speed_1_control']

# if vergence_dfs:
#     print(f"\nFound {len(vergence_dfs)} dataframes with vergence in name")
    
#     # Calculate average eating events per episode for each dataframe
#     eating_stats_vergence = {}
#     for name, df in vergence_dfs.items():
#         print(df.keys())
#         # Group by episode and count eating events per episode
#         eating_per_episode = df.groupby(['env_id', 'episode_index'])['eating_event'].sum()
#         avg_eating_per_episode = eating_per_episode.mean()
#         std_eating_per_episode = eating_per_episode.std() / (np.sqrt(len(eating_per_episode)))
#         print(f"{name}: {avg_eating_per_episode:.2f} ± {std_eating_per_episode:.2f} eating events per episode")

#         # Get cumulative reward at final time step for each episode
#         final_rewards_per_episode = df.groupby(['env_id', 'episode_index'])['cumulative_reward'].last()
#         avg_final_reward = final_rewards_per_episode.mean()
#         std_final_reward = final_rewards_per_episode.std() / (np.sqrt(len(final_rewards_per_episode)))
#         print(f"{name}: {avg_final_reward:.2f} ± {std_final_reward:.2f} average final cumulative reward per episode")

#         eating_stats_vergence[name] = (avg_eating_per_episode, std_eating_per_episode, avg_final_reward, std_final_reward)
    
#     # Create bar plot
#     plt.figure(figsize=(12, 6))
    
#     names = list(eating_stats_vergence.keys())
#     eating_values_vergence = [stats[0] for stats in eating_stats_vergence.values()]
#     eating_std_vergence = [stats[1] for stats in eating_stats_vergence.values()]
#     reward_values_vergence = [stats[2] for stats in eating_stats_vergence.values()]
    
#     x = np.arange(len(names))
#     width = 0.35
    
#     fig, ax1 = plt.subplots(figsize=(12, 6))
    
#     # Create bars for eating events with error bars
#     bars1 = ax1.bar(x - width/2, eating_values_vergence, width, yerr=eating_std_vergence, 
#                     label='Eating Events', alpha=0.7, capsize=5)
#     ax1.set_xlabel('Vergence Configuration')
#     ax1.set_ylabel('Average Eating Events per Episode', color='tab:blue')
#     ax1.tick_params(axis='y', labelcolor='tab:blue')
#     ax1.set_xticks(x)
#     ax1.set_xticklabels(names, rotation=45, ha='right')
    
#     # Create second y-axis for rewards
#     # ax2 = ax1.twinx()
#     # bars2 = ax2.bar(x + width/2, reward_values_vergence, width, label='Cumulative Reward', alpha=0.7, color='orange')
#     # ax2.set_ylabel('Average Final Cumulative Reward per Episode', color='tab:orange')
#     # ax2.tick_params(axis='y', labelcolor='tab:orange')
    
#     plt.title('Average Eating Events per Episode vs Vergence Configuration')
#     plt.tight_layout()
    
#     # Create plots directory and save
#     plots_dir = os.path.join(outputs_folder, "additional_exps", "plots")
#     os.makedirs(plots_dir, exist_ok=True)
    
#     plot_path = os.path.join(plots_dir, "eating_events_vs_vergence.png")
#     plt.savefig(plot_path, dpi=300, bbox_inches='tight')
#     print(f"\nBar plot saved to: {plot_path}")
#     plt.show()
# else:
#     print("\nNo dataframes found with vergence in name")
