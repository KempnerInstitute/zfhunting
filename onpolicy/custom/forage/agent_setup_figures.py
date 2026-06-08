# %%
import numpy as np
import matplotlib.pyplot as plt

# Compute the function values
x = np.linspace(-40 * np.pi/180, 40 * np.pi/180, 1000)
y = np.exp(-10 * np.abs(x))

# Plot the function
plt.figure(figsize=(10, 6))
plt.plot(x, y, color='darkblue', linewidth=2)

# Add decorations
plt.title(r'Eating Probability vs. Orientation to Food', fontsize=16)
plt.xlabel(r'$\mathrm{\theta_{eating}}$', fontsize=14)
plt.ylabel('Eating Probability', fontsize=14)
plt.grid(alpha=0.3)

# Relabel x-axis in degrees
x_ticks = np.linspace(-40 * np.pi/180, 40 * np.pi/180, 5)  # Define tick positions in radians
x_labels = [f"{np.degrees(tick):.0f}°" for tick in x_ticks]  # Convert to degrees
plt.xticks(ticks=x_ticks, labels=x_labels)

plt.tight_layout()

# Show the plot
plt.show()


# %%
import seaborn as sns

# Set Seaborn style
sns.set(style="whitegrid")

# Plot the function with Seaborn styling
plt.figure(figsize=(10, 6))
plt.plot(x, y, color=sns.color_palette("muted")[0], linewidth=2)

# Add decorations
# plt.title(r'Eating Probability vs. Orientation to Food', fontsize=16)
# plt.xlabel(r'$\mathrm{\theta_{eating}}$', fontsize=20)
# plt.ylabel('Eating Probability', fontsize=20)
plt.grid(True, alpha=0.3, linestyle='--')

# Relabel x-axis in degrees
plt.xticks(ticks=x_ticks, labels=x_labels, fontsize=30)
plt.yticks(fontsize=30)

# Adjust x-axis limits to make the plot more condensed
plt.xlim(x_ticks[0], x_ticks[-1])
plt.ylim(-0.0, 1.0)

plt.tight_layout()

# Show the plot
plt.savefig("eating_probability_vs_orientation.svg", format='svg')

plt.show()


# %%
import numpy as np
import matplotlib.pyplot as plt

# Create a grid of possible actions
move_actions = np.linspace(0, 1, 100)
turn_actions = np.linspace(-1, 1, 100)
M, T = np.meshgrid(move_actions, turn_actions)

# Apply the transformation
T_transformed = T * (1 - M)

# Plot the action space
plt.figure(figsize=(8, 6))
plt.scatter(M, T_transformed, c=T_transformed, cmap='coolwarm', s=5)
plt.xlabel("move_action")
plt.ylabel("turn_action (transformed)")
plt.title("Action Space after Transformation: turn_action * (1 - move_action)")
plt.colorbar(label="Turn Action Value")
plt.grid(True)
plt.show()


# %%
import cfg

# Define the vertices of the triangle
triangle_vertices = [(0, 1), (0, -1), (1, 0)]

# Extract x and y coordinates of the vertices
triangle_x, triangle_y = zip(*triangle_vertices)

# Close the triangle by appending the first vertex at the end
triangle_x += (triangle_x[0],)
triangle_y += (triangle_y[0],)

# Plot the triangle
plt.figure(figsize=(6, 6))
plt.plot(triangle_x, triangle_y, color='black', linewidth=2)
plt.fill(triangle_x, triangle_y, color='lightblue', alpha=0.5)

# Set axis limits
plt.xlim(-0.005, 1)
plt.ylim(-1, 1)

# Relabel x and y axes by scaling them
x_ticks = np.linspace(0, 1, 3)  # Define tick positions in the range [0, 1]
x_ticks_scaled = [tick * cfg.FISH_CONSTANTS["max_speed"] for tick in x_ticks]
y_ticks = np.linspace(-1, 1, 5)  # Define fewer tick positions
y_ticks_scaled = [tick * cfg.FISH_CONSTANTS["max_turn_speed"] for tick in y_ticks]
plt.yticks(ticks=y_ticks, labels=[f"{tick:.1f}" for tick in y_ticks_scaled], fontsize=30)

plt.xticks(ticks=x_ticks, labels=[f"{tick:.2f}" for tick in x_ticks_scaled], fontsize=30)
plt.yticks(ticks=plt.yticks()[0], labels=[f"{tick:.2f}" for tick in y_ticks_scaled], fontsize=30)

# Add labels and title
#plt.xlabel('Forward Speed (mm/s)', fontsize=30)
#plt.ylabel('Turn Speed (rad/s)', fontsize=30)

# Show grid and plot
plt.tight_layout()
plt.grid(True, alpha=0.3, linestyle='--')
plt.savefig("action_space_triangle.svg", format='svg')
plt.show()

# %%
import numpy as np
import matplotlib.pyplot as plt

# Sample 100 points within the action space
num_samples = 10000000
action_noise = 0.1
x_samples = np.random.uniform(0, 5, num_samples)
y_samples = np.random.uniform(-7 * (1 - x_samples/5), 7 * (1 - x_samples/5), num_samples)

# Scale x and y by 0.2
x_scaled = x_samples * 0.2
y_scaled = y_samples * 0.2

# Combine x and y into a list of points
sampled_points = list(zip(x_scaled, y_scaled))

print(sampled_points)
# Initialize lists to store all trajectories
all_trajectories = []

# Compute each trajectory
for move_forward, turn in sampled_points:
    # Define the starting location for the current trajectory
    start_x, start_y = 0, 0
    trajectory_x = [start_x]
    trajectory_y = [start_y]
    
    # Update the direction based on the turn value
    direction = turn

    move_forward = move_forward * (1 + np.random.uniform(-action_noise*np.sqrt(12)/2, action_noise*np.sqrt(12)/2))
    turn = turn * (1 + np.random.uniform(-action_noise*np.sqrt(12)/2, action_noise*np.sqrt(12)/2))

    # Compute the new position
    new_x = trajectory_x[-1] + move_forward * np.cos(direction)
    new_y = trajectory_y[-1] + move_forward * np.sin(direction)
    
    # Append the new position to the trajectory
    trajectory_x.append(new_x)
    trajectory_y.append(new_y)
    
    # Store the trajectory
    all_trajectories.append((trajectory_x, trajectory_y))

# Plot all trajectories
# plt.figure(figsize=(10, 6))
# for trajectory_x, trajectory_y in all_trajectories:
#     plt.plot(trajectory_x, trajectory_y, marker='o', markersize=2, linestyle='-', alpha=0.7)

# plt.title("Trajectories Plot", fontsize=16)
# plt.xlabel("X Position", fontsize=14)
# plt.ylabel("Y Position", fontsize=14)
# plt.grid(alpha=0.3)
# plt.axis('equal')
# plt.show()


# %%
# # Plot all trajectories
# plt.figure(figsize=(10, 6))
# for trajectory_x, trajectory_y in all_trajectories:
#     plt.plot(trajectory_x, trajectory_y, marker='o', markersize=2, linestyle='-', alpha=0.7)

# plt.title("Trajectories Plot", fontsize=16)
# plt.xlabel("X Position", fontsize=14)
# plt.ylabel("Y Position", fontsize=14)
# plt.grid(alpha=0.3)
# plt.axis('equal')
# plt.show()

# Extract end locations of trajectories
end_locations_x = [trajectory_x[-1] for trajectory_x, _ in all_trajectories]
end_locations_y = [trajectory_y[-1] for _, trajectory_y in all_trajectories]

# Plot the distribution of end locations
# plt.figure(figsize=(10, 6))
# plt.scatter(end_locations_x, end_locations_y, alpha=0.7, c='blue', s=10)
# plt.title("Distribution of End Locations", fontsize=16)
# plt.xlabel("X Position", fontsize=14)
# plt.ylabel("Y Position", fontsize=14)
# plt.grid(alpha=0.3)
# plt.axis('equal')
# plt.show()


# %%
from matplotlib.colors import LogNorm

# Create a figure with a larger size for better visibility
plt.figure(figsize=(12, 8))

# Plot the hexbin heatmap with improved aesthetics
hexbin_plot = plt.hexbin(
    end_locations_x, 
    end_locations_y, 
    gridsize=100, 
    cmap='viridis',  # Use a bluish colormap
    mincnt=1, 
    norm=LogNorm()
)

# Add a colorbar with a more descriptive label
cbar = plt.colorbar(hexbin_plot, label='Density (log scale)', pad=0.02)
cbar.ax.tick_params(labelsize=14)  # Increase colorbar tick label size

# Add title and axis labels with larger font sizes
plt.title("Heatmap of End Locations (Logarithmic Scale)", fontsize=20, pad=20)
plt.xlabel("X Position", fontsize=16, labelpad=10)
plt.ylabel("Y Position", fontsize=16, labelpad=10)

# Customize tick parameters for better readability
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

# Add a grid for better visual alignment
# plt.grid(alpha=0.3, linestyle='--')

# Ensure the aspect ratio is equal for accurate representation
plt.axis('equal')

# Save the plot with high resolution for publication
# plt.savefig("heatmap_end_locations_final.png", dpi=300, bbox_inches='tight')

# Show the plot
plt.tight_layout()
plt.savefig("heatmap_end_locations_final.svg", format='svg')
plt.show()

# %%
from scipy.stats import gaussian_kde
import numpy as np

# Perform kernel density estimation
xy = np.vstack([end_locations_x, end_locations_y])
kde = gaussian_kde(xy, bw_method='scott')  # Adjust bandwidth if needed
x_min, x_max = min(end_locations_x), max(end_locations_x)
y_min, y_max = min(end_locations_y), max(end_locations_y)

# Create a grid for evaluation
x_grid, y_grid = np.linspace(x_min, x_max, 500), np.linspace(y_min, y_max, 500)
X, Y = np.meshgrid(x_grid, y_grid)
Z = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)

# Plot the smoothed density
plt.figure(figsize=(12, 8))
plt.contourf(X, Y, Z, levels=50, cmap='viridis')  # Use filled contours for smoothness
cbar = plt.colorbar(label='Density')
cbar.ax.tick_params(labelsize=14)

# Add title and axis labels
plt.title("Smoothed Heatmap of End Locations", fontsize=20, pad=20)
plt.xlabel("X Position", fontsize=16, labelpad=10)
plt.ylabel("Y Position", fontsize=16, labelpad=10)

# Customize ticks
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

# Ensure equal aspect ratio
plt.axis('equal')
plt.tight_layout()
plt.savefig("smoothed_heatmap_end_locations.svg", format='svg')
plt.show()


# %%



