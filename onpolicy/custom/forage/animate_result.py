#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import imageio
import os
from matplotlib.animation import FuncAnimation
from datetime import datetime
from moviepy.editor import VideoFileClip, clips_array, vfx
import matplotlib.patches as mpatches  # For creating custom legends
import matplotlib.gridspec as gridspec
from sklearn.preprocessing import StandardScaler
from scipy.cluster import hierarchy as sch
import seaborn as sns


# Define the OBJECT_TYPES mapping
OBJECT_TYPES = {
    "WALL": 2.0,
    "FOOD": 0.5,
    "AGENT": 1.0,
    "NONE": 0.0,
}

# Define colors for observation cases
colors = {
    0: 'grey',     # Neither food nor agent
    1: 'green',    # Food only
    2: 'orange',   # Agent only
    3: 'blue',     # Both food and agent
}

# Read the pickled DataFrame
flat_pkl_file = "./results/20241008/rmappo-MultiAgentForagingEnv-replenish_20241008_11_sr20_trainsteps1000000_layerN1_hs64_seed1/rollouts_20241008_114722_4agents_N6GQkLQn_4_flattened.pkl"
dff = pd.read_pickle(flat_pkl_file)

mother_folder = os.path.dirname(flat_pkl_file)

# Replace the string after "agents" with "_rnn_reps_plots/"
rnn_anim_folder = os.path.join(flat_pkl_file.split("agents")[0] + "agents_rnn_anims_/")

# Create the directory if it doesn't exist
os.makedirs(rnn_anim_folder, exist_ok=True)
print(f"Saving anims to: {rnn_anim_folder}")

# Squeeze the RNN states if necessary
dff['rnn_states'] = dff['rnn_states'].apply(lambda x: x.squeeze())

# Forward-fill and backward-fill 'food_positions' to ensure all rows have it
dff['food_positions'] = dff.groupby(['env_id', 'episode_index'])['food_positions'].transform(
    lambda x: x.ffill().bfill()
)

# Sort the DataFrame
dff = dff.sort_values(by=['env_id', 'episode_index', 'time_step', 'agent_id'])

# Get unique episodes
episodes = dff[['env_id', 'episode_index']].drop_duplicates()

# Set arena size (adjust based on your environment settings)
arena_size = (100, 100)

# Output directory for videos
output_dir = './videos'
os.makedirs(output_dir, exist_ok=True)

# Tail length for traces
tail_length = 10

def compute_observation_case(observation):
    object_types = observation[1::2]
    has_food = OBJECT_TYPES["FOOD"] in object_types
    has_agent = OBJECT_TYPES["AGENT"] in object_types
    if has_food and has_agent:
        return 3  # Both food and agent
    elif has_food:
        return 1  # Food only
    elif has_agent:
        return 2  # Agent only
    else:
        return 0  # Neither food nor agent


import matplotlib.gridspec as gridspec
from sklearn.preprocessing import StandardScaler
from scipy.cluster import hierarchy as sch
from mpl_toolkits.axes_grid1 import make_axes_locatable

def animate_agent_episode(agent_df, positions_df, food_positions_df, pca_agent_df, agent_id, env_id, episode_index):
    """Creates an animation for a single agent in a single episode."""
    # Initialize figure with subplots using GridSpec
    fig = plt.figure(figsize=(12, 8))
    gs = gridspec.GridSpec(2, 3, height_ratios=[1, 1], width_ratios=[2, 2, 1])

    # First row
    ax_env = fig.add_subplot(gs[0, 0])
    ax_rnn = fig.add_subplot(gs[0, 1])
    ax_legend = fig.add_subplot(gs[0, 2])

    # Second row
    ax_actions = fig.add_subplot(gs[1, 0])
    ax_obs_heatmap = fig.add_subplot(gs[1, 1])
    ax_neural_heatmap = fig.add_subplot(gs[1, 2])

    # Number of frames
    num_frames = len(agent_df)

    # Plot the background scatter plot of agent's PCA projections
    for case, group in pca_agent_df.groupby('obs_case'):
        ax_rnn.scatter(group['pc1'], group['pc2'], color=colors[case], alpha=0.4, s=4)

    ax_rnn.set_xlabel('PC1')
    ax_rnn.set_ylabel('PC2')
    ax_rnn.set_title(f'Agent {agent_id} RNN State Trajectory')

    # Hide the axis for the legend subplot
    ax_legend.axis('off')

    # Create a custom legend for the different observation cases in the legend subplot
    legend_elements = [
        mpatches.Patch(color=colors[0], label='Neither food nor agent'),
        mpatches.Patch(color=colors[1], label='Food only'),
        mpatches.Patch(color=colors[2], label='Agent only'),
        mpatches.Patch(color=colors[3], label='Both food and agent')
    ]
    ax_legend.legend(handles=legend_elements, loc='center left')  # Add the legend to the separate subplot

    # Prepare data for plotting
    # Positions per timestep
    positions_per_timestep = positions_df.groupby('time_step').apply(
        lambda x: dict(zip(x['agent_id'], x['position']))
    ).to_dict()

    # Food positions per timestep
    food_positions_per_timestep = food_positions_df.set_index('time_step')['food_positions'].to_dict()

    # Positions over time for each agent
    positions_per_agent = positions_df.groupby('agent_id').apply(
        lambda x: x.set_index('time_step')['position'].to_dict()
    ).to_dict()

    # Process actions
    agent_df['move_forward_processed'] = agent_df['move_forward'].apply(lambda x: 2 * 1 / (1 + np.exp(-x)))
    agent_df['turn_angle_processed'] = agent_df['turn_angle'].apply(lambda x: np.tanh(x) * 45)

    # Process observations
    agent_df['obs_distances'] = agent_df['observations'].apply(lambda obs: obs[::2])
    agent_df['obs_object_types'] = agent_df['observations'].apply(lambda obs: obs[1::2])

    # Prepare RNN states for clustering
    rnn_states_agent = np.vstack(agent_df['rnn_states'].tolist())
    X = rnn_states_agent  # Shape: (n_time_steps, n_neurons)
    X_n = StandardScaler().fit_transform(X)  # Normalize the data

    # Compute covariance matrix and hierarchical clustering
    matrix = np.cov(X_n.T)
    linkage_matrix = sch.linkage(matrix, method='ward')
    dendro = sch.dendrogram(linkage_matrix, no_plot=True)
    order = dendro['leaves']

    # Initialize tail and current point for RNN plot
    tail_scatter = ax_rnn.scatter([], [], s=10)
    current_point_scatter = ax_rnn.scatter([], [], s=50)
    rnn_trace_line, = ax_rnn.plot([], [], color='gray', lw=1)  # Line connecting the tail points

    # Initialize the time series plot for actions
    ax_actions2 = ax_actions.twinx()  # Create a twin y-axis

    line_move_forward, = ax_actions.plot([], [], label='Move Forward', color='blue')
    line_turn_angle, = ax_actions2.plot([], [], label='Turn Angle', color='red')

    ax_actions.set_ylim(0, 2)
    ax_actions2.set_ylim(-45, 45)

    ax_actions.set_ylabel('Move Forward')
    ax_actions2.set_ylabel('Turn Angle')

    # Adjust legends
    lines = [line_move_forward, line_turn_angle]
    labels = [line.get_label() for line in lines]
    ax_actions.legend(lines, labels, loc='upper left')

    ax_actions.set_title('Actions over Time')
    ax_actions.set_xlabel('Time Step')

    # Initialize observations RGBA array
    num_rays = 30
    obs_rgba_data = np.zeros((num_rays, tail_length, 4))
    obs_heatmap = ax_obs_heatmap.imshow(obs_rgba_data, aspect='auto', origin='lower', interpolation='nearest')
    ax_obs_heatmap.set_title('Observations')
    ax_obs_heatmap.set_xlabel('Time Step')
    ax_obs_heatmap.set_ylabel('Ray Index')

    # Create custom legend for observations
    obs_legend_elements = [
        mpatches.Patch(color='red', label='Food'),
        mpatches.Patch(color='blue', label='Agent'),
        mpatches.Patch(color='yellow', label='Wall')
    ]
    ax_obs_heatmap.legend(handles=obs_legend_elements, loc='upper left')

    # Initialize neural activities heatmap
    neural_heatmap_data = np.zeros((X_n.shape[1], tail_length))
    neural_heatmap = ax_neural_heatmap.imshow(neural_heatmap_data, aspect='auto', origin='lower', interpolation='nearest', cmap='bwr')
    ax_neural_heatmap.set_title('Neural Activities')
    ax_neural_heatmap.set_xlabel('Time Step')
    ax_neural_heatmap.set_ylabel('Neuron Index')

    # Add colorbar for neural activities
    divider = make_axes_locatable(ax_neural_heatmap)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    plt.colorbar(neural_heatmap, cax=cax)

    # Define colors for object types
    object_colors = {
        OBJECT_TYPES["FOOD"]: np.array([1, 0, 0]),  # Red
        OBJECT_TYPES["AGENT"]: np.array([0, 0, 1]),  # Blue
        OBJECT_TYPES["WALL"]: np.array([1, 1, 0]),   # Yellow
        OBJECT_TYPES["NONE"]: np.array([0.5, 0.5, 0.5])  # Grey
    }

    # Frames for animation
    frames = []

    for frame in range(num_frames):
        time_step = agent_df.iloc[frame]['time_step']

        # Clear environment axis
        ax_env.clear()

        # Update environment subplot
        # Plot the tail of positions for each agent
        for aid in positions_per_agent.keys():
            positions = positions_per_agent[aid]
            times = sorted([t for t in positions.keys() if t <= time_step])
            if len(times) > 0:
                last_times = times[-tail_length:]
                pos_list = [positions[t] for t in last_times if positions[t] is not None]
                if len(pos_list) > 0:
                    pos_array = np.array(pos_list)
                    color = 'blue' if aid == agent_id else 'red'
                    ax_env.plot(pos_array[:, 0], pos_array[:, 1], '-', color=color, linewidth=1)
                    ax_env.plot(pos_array[-1, 0], pos_array[-1, 1], 'o', color=color)
                    ax_env.text(pos_array[-1, 0], pos_array[-1, 1], str(aid), ha='center', va='center', fontsize=7, color='white')

        # Plot food positions
        food_positions = food_positions_per_timestep.get(time_step, [])
        if food_positions is not None and len(food_positions) > 0:
            food_positions_array = np.array(food_positions)
            ax_env.plot(food_positions_array[:, 0], food_positions_array[:, 1], 'go', markersize=3)

        ax_env.set_xlim(0, arena_size[0])
        ax_env.set_ylim(0, arena_size[1])
        ax_env.set_aspect('equal')
        ax_env.set_title(f'Environment at Time Step {time_step}')

        # Update RNN state subplot
        # Plot the tail
        tail_start = max(0, frame - tail_length + 1)
        tail_data = pca_agent_df.iloc[tail_start:frame+1]

        x_tail = tail_data['pc1'].values
        y_tail = tail_data['pc2'].values
        colors_tail = tail_data['obs_case'].map(colors).values

        # Update scatter for the tail
        tail_scatter.set_offsets(np.column_stack([x_tail, y_tail]))
        tail_scatter.set_color(colors_tail)

        # Update line for the tail trace
        rnn_trace_line.set_data(x_tail, y_tail)

        # Update current point
        x_current = pca_agent_df.iloc[frame]['pc1']
        y_current = pca_agent_df.iloc[frame]['pc2']
        color_current = colors[pca_agent_df.iloc[frame]['obs_case']]

        current_point_scatter.set_offsets(np.array([[x_current, y_current]]))
        current_point_scatter.set_color(color_current)

        # Update actions plot
        action_tail_data = agent_df.iloc[tail_start:frame+1]
        time_steps = action_tail_data['time_step'].values
        move_forward_values = action_tail_data['move_forward_processed'].values
        turn_angle_values = action_tail_data['turn_angle_processed'].values

        line_move_forward.set_data(time_steps, move_forward_values)
        line_turn_angle.set_data(time_steps, turn_angle_values)

        ax_actions.set_xlim(time_steps[0], time_steps[-1])
        ax_actions2.set_xlim(time_steps[0], time_steps[-1])

        ax_actions.set_ylim(0, 2)
        ax_actions2.set_ylim(-45, 45)

        # Update observations heatmap
        obs_distances_list = agent_df.iloc[tail_start:frame+1]['obs_distances'].tolist()
        obs_object_types_list = agent_df.iloc[tail_start:frame+1]['obs_object_types'].tolist()

        num_time_steps = len(obs_distances_list)
        obs_rgba_data = np.zeros((num_rays, num_time_steps, 4))  # Shape (num_rays, num_time_steps, 4)

        for i, (distances, object_types) in enumerate(zip(obs_distances_list, obs_object_types_list)):
            for ray in range(num_rays):
                obj_type = object_types[ray]
                norm_distance = distances[ray]  # x in [0,1]
                alpha = 1 - 0.95 * norm_distance  # Closer objects have higher alpha

                if obj_type in [OBJECT_TYPES["FOOD"], OBJECT_TYPES["AGENT"], OBJECT_TYPES["WALL"]]:
                    color = object_colors[obj_type]
                    obs_rgba_data[ray, i, :3] = color  # Set RGB
                    obs_rgba_data[ray, i, 3] = alpha  # Set alpha
                else:
                    # For NONE, leave as zeros (transparent)
                    pass

        obs_heatmap.set_data(obs_rgba_data)
        obs_heatmap.set_extent([time_steps[0], time_steps[-1], 0, num_rays])

        ax_obs_heatmap.set_xlim(time_steps[0], time_steps[-1])
        ax_obs_heatmap.set_ylim(0, num_rays)
        ax_obs_heatmap.set_aspect('auto')

        # Update neural activities heatmap
        neural_activities = X_n[tail_start:frame+1, :]  # Shape: (num_time_steps, n_neurons)
        ordered_neural_activities = neural_activities[:, order].T  # Shape: (n_neurons, num_time_steps)

        neural_heatmap.set_data(ordered_neural_activities)
        neural_heatmap.set_extent([time_steps[0], time_steps[-1], 0, ordered_neural_activities.shape[0]])
        neural_heatmap.set_clim(-3, 3)  # Adjust based on your data

        ax_neural_heatmap.set_xlim(time_steps[0], time_steps[-1])
        ax_neural_heatmap.set_ylim(0, ordered_neural_activities.shape[0])
        ax_neural_heatmap.set_aspect('auto')

        # Draw the canvas and convert to image
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(image)

    plt.close(fig)

    # Save the frames as a video (slower animation with fps=5)
    output_filename = f'{rnn_anim_folder}/agent_{agent_id}_env_{env_id}_ep_{episode_index}.mp4'
    imageio.mimsave(output_filename, frames, fps=5)  # Slowing down the animation
    print(f'Saved video: {output_filename}')

    return output_filename




# Loop through each episode and agent to create videos
for env_id in dff['env_id'].unique()[:1]:
    for episode_index in dff['episode_index'].unique()[:1]:
        episode_df = dff[(dff['env_id'] == env_id) & (dff['episode_index'] == episode_index)]

        agent_ids = episode_df['agent_id'].unique()

        # Positions DataFrame for the episode
        positions_df = episode_df[['time_step', 'agent_id', 'position']].dropna()
        # Food positions DataFrame for the episode
        food_positions_df = episode_df[['time_step', 'food_positions']].drop_duplicates('time_step').dropna()

        video_files = []

        for agent_id in agent_ids:
            agent_df = episode_df[episode_df['agent_id'] == agent_id].sort_values('time_step')
            if agent_df.empty:
                continue

            # Compute observation cases for the agent
            agent_df['obs_case'] = agent_df['observations'].apply(compute_observation_case)

            # Get RNN states and PCA projections for the agent
            rnn_states_agent = np.vstack(agent_df['rnn_states'].tolist())
            # Fit PCA on the agent's RNN states
            pca_agent = PCA(n_components=2)
            pca_agent.fit(rnn_states_agent)
            pca_result_agent = pca_agent.transform(rnn_states_agent)

            # Create DataFrame with PCA projections and observation cases
            pca_agent_df = agent_df[['time_step', 'obs_case']].copy()
            pca_agent_df['pc1'] = pca_result_agent[:, 0]
            pca_agent_df['pc2'] = pca_result_agent[:, 1]

            # Create animation for the agent
            video_file = animate_agent_episode(
                agent_df, positions_df, food_positions_df, pca_agent_df, agent_id, env_id, episode_index
            )
            video_files.append(video_file)

        # Optionally, create a collage of agent videos per episode
        def create_collage(video_files, output_filename):
            clips = [VideoFileClip(video) for video in video_files]
            num_videos = len(clips)
            num_columns = 2
            num_rows = (num_videos + 1) // 2

            # Add blank clips if necessary to fill the grid
            while len(clips) < num_columns * num_rows:
                duration = clips[0].duration
                size = clips[0].size
                blank_clip = VideoFileClip(video_files[0]).fx(vfx.colorx, 0).set_duration(duration).resize(size)
                clips.append(blank_clip)

            # Arrange clips in a grid
            clips_grid = [clips[i * num_columns:(i + 1) * num_columns] for i in range(num_rows)]
            final_clip = clips_array(clips_grid)

            # Save the collage video
            final_clip.write_videofile(output_filename, fps=24)
            print(f'Saved collage video to {output_filename}')

        # Create a collage per episode (if desired)
        if len(video_files) > 0:
            collage_output = f'{rnn_anim_folder}/collage_env_{env_id}_ep_{episode_index}.mp4'
            create_collage(video_files, collage_output)