import numpy as np
import gym
from gym import spaces
from gym.envs.registration import EnvSpec
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import io
import os
from PIL import Image
import imageio
import tqdm
from scipy.spatial import distance_matrix
from scipy.spatial.distance import cdist
from datetime import datetime
import string
import random
import matplotlib.patheffects as path_effects
from matplotlib.patches import Arc
import arena as ar
from matplotlib.patches import Wedge
import cfg

OBJECT_TYPES = cfg.OBJECT_TYPES
FISH_CONSTANTS = cfg.FISH_CONSTANTS
ENV_PARAMS = cfg.ENV_PARAMS
AGENT_PARAMS = cfg.AGENT_PARAMS
REWARDS = cfg.REWARDS

UNKNOWN_DIST = -1.0


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def polar_to_cartesian(r, theta):
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y


def relative_to_global(position_polar, reference_position, reference_orientation):
    distance, angle = position_polar
    if distance < 0:  # Skip dummy observations
        return None
    x, y = polar_to_cartesian(distance, angle + reference_orientation)
    return reference_position + np.array([x, y])


def transform_to_relative(position, reference_position, reference_orientation):
    relative_position = position - reference_position
    rotation_matrix = np.array(
        [
            [np.cos(reference_orientation), -np.sin(reference_orientation)],
            [np.sin(reference_orientation), np.cos(reference_orientation)],
        ]
    )
    return np.dot(relative_position, rotation_matrix.T)


def cartesian_to_polar(xy):
    x, y = xy
    distance = np.sqrt(x**2 + y**2)
    angle = np.arctan2(y, x)
    return distance, angle


def normalize_angle(theta):
    """Normalize angle to be within [-pi, pi]."""
    return (theta + np.pi) % (2 * np.pi) - np.pi


class ZFishAgent:
    def __init__(
        self,
        all_args,
        arena_size,
        max_step,
        max_turn,
        max_perception_turn,
        agent_id,
        energy_minimum,
        eating_dist_decay,
        seed=None,
    ):
        self.arena_size = arena_size
        self.max_step = max_step
        self.max_turn = max_turn
        self.perception_field = FISH_CONSTANTS["perception_field"]
        self.max_perception_turn = max_perception_turn
        self.max_left_vergence = FISH_CONSTANTS["max_left_vergence"]
        self.max_right_vergence = FISH_CONSTANTS["max_right_vergence"]
        self.min_left_vergence = FISH_CONSTANTS["min_left_vergence"]
        self.min_right_vergence = FISH_CONSTANTS["min_right_vergence"]
        self.agent_id = agent_id
        self.energy_minimum = energy_minimum
        self.food_sensing_radius = all_args["food_detection_range"]
        self.fish_sensing_radius = all_args["fish_detection_range"]
        self.wall_sensing_radius = all_args["sensing_radius"] #maybe this should be changed to the max of the other sensing radii
        self.energy_per_step = AGENT_PARAMS[
            "energy_per_step"
        ]  # scaled by move_forward
        self.energy_food = AGENT_PARAMS["energy_food"]
        self.num_rays = AGENT_PARAMS["num_rays"]
        self.ray_length = all_args["sensing_radius"] #maybe this should be changed to the max of the other sensing radii
        self.eating_angle = all_args["eating_angle"]
        self.baseline_success_eating_angle = all_args.get(
            "baseline_success_eating_angle", cfg.AGENT_PARAMS["baseline_success_eating_angle"]
        )
        self.eating_distribution_decay = eating_dist_decay
        self.eating_radius = AGENT_PARAMS["eating_radius"]
        self.food_radius = ENV_PARAMS["food_radius"]
        self.agent_radius = AGENT_PARAMS["body_radius"]
        self.distance_noise_std = (
            AGENT_PARAMS["distance_noise_std"] * np.sqrt(12) / 2
        )  # Standard deviation of noise added to distance measurement
        self.detection_failure_rate = all_args[
            "detection_failure_rate"
        ]  # Probability of complete detection failure
        self.false_positive_rate = all_args[
            "false_positive_rate"
        ]  # Probability of false positive detection
        self.angle_noise_std_food = all_args.get("angle_noise_std_food", 0) * np.sqrt(12) / 2
        self.angle_noise_std_walker = all_args.get("angle_noise_std_walker", 0) * np.sqrt(12) / 2
        self.collided = False
        self.curr_food_consumed = []
        self.max_food_eaten_per_step = all_args["max_food_eaten_per_step"]
        self.num_actions = 4
        self.last_action = [0] * self.num_actions
        self.body_radius = AGENT_PARAMS["body_radius"]
        self.has_nearby = True
        self.energy_maximum = AGENT_PARAMS["energy_maximum"]
        self.energy_minimum = AGENT_PARAMS["energy_minimum"]
        self.discrete_actions = all_args["discrete_actions"]

        self.num_sensors = self.num_rays
        self.projected_sector_angle = self.perception_field / self.num_sensors
        self.sensed_agents = []
        self.left_eye_sensors = np.zeros((self.num_sensors, 2))
        self.right_eye_sensors = np.zeros((self.num_sensors, 2))

        self.binocular_depth_only = all_args["binocular_depth_only"]
        self.binocular_angle_only = all_args["binocular_angle_only"]
        self.binary_eye_state = all_args["binary_eye_state"]
        self.flash_monocular_only = all_args["flash_monocular_only"]
        self.conspecific_monocular_perception = all_args.get(
            "conspecific_monocular_perception", True
        )

        self.eye_separation = FISH_CONSTANTS["eye_separation"]
        self.eye_forward_offset = FISH_CONSTANTS["eye_forward_offset"]
        self.eye_hypotenuse = FISH_CONSTANTS["eye_hypotenuse"]
        self.use_1dof_eyes = all_args["use_1dof_eyes"]

        self.use_fatigue = AGENT_PARAMS["fatigue"]
        self.fatigue_recovery = AGENT_PARAMS["fatigue_recovery"]
        self.eye_fatigue_recovery = all_args["eye_fatigue_recovery"]
        self.forced_interbout = AGENT_PARAMS["forced_interbout"]

        self.eye_persistence = all_args["eye_persistence"]
        self.use_eye_fatigue = all_args["use_eye_fatigue"]
        self.eye_fatigue_penalty = all_args.get("eye_fatigue_penalty", cfg.REWARDS["eye_fatigue_penalty"])

        self.action_noise_std = (
            all_args.get("action_noise_std", AGENT_PARAMS["action_noise_std"])
            * np.sqrt(12)
            / 2
        )  # Convert to uniform noise range [-action_noise_std, action_noise_std]

        self.eye_muscle_model = all_args["eye_muscle_model"]
        self.k_relax_eye = all_args["k_relax_eye"]
        self.g_input_eye = all_args["g_input_eye"]

        self.arena_radius = None
        self.arena_center = None

        self.move_forward_values = np.concatenate(
            [[0], np.logspace(-1, 0, AGENT_PARAMS["num_speeds"])]
        )
        self.turn_angle_values = np.linspace(-1, 1, AGENT_PARAMS["num_turn_angles"])

        # self.reset(seed=seed)

    def reset(self, seed=None):
        if seed is not None:
            self.np_random = np.random.default_rng(seed=seed)
        elif not hasattr(self, "np_random"):
            self.np_random = np.random.default_rng()

        # Sample position uniformly within circular arena
        # For circular arena, sample uniformly within circle
        # Use polar coordinates
        angle = self.np_random.uniform(0, 2 * np.pi)
        # Use sqrt for uniform distribution within circle
        arena_radius = self.arena_radius
        radius = np.sqrt(self.np_random.uniform(0, 1)) * (
            arena_radius - self.body_radius
        )
        center = np.array([arena_radius, arena_radius])
        self.position = center + radius * np.array([np.cos(angle), np.sin(angle)])
        self.orientation = self.np_random.uniform(-np.pi, np.pi)
        self.left_eye_angle = (
            self.min_left_vergence
        )  # relative to orientation, initialize eyes in rest position
        self.right_eye_angle = (
            self.min_right_vergence
        )  # relative to orientation, initialize eyes in rest position
        self.energy = self.np_random.uniform(self.energy_minimum + 30, 100)
        self.trajectory = [self.position.copy()]
        self.move_forward = 0
        self.turn_angle = 0
        self.turn_left_eye = 0
        self.turn_right_eye = 0
        self.cumulative_reward = 0
        self.previous_food_distance = None
        self.collided = False
        self.curr_food_consumed = []
        self.left_eye_sensors = np.zeros((self.num_sensors, 2))
        self.right_eye_sensors = np.zeros((self.num_sensors, 2))
        self.left_detections = {}
        self.right_detections = {}
        self.food_consumed_ids = []
        self.detected_food_ids = []
        self.fatigue_count = 0
        self.eye_fatigue = 0
        self.interbout_wait_time = 1
        self.eye_wait_time = self.eye_persistence

        if self.binary_eye_state:
            self.left_eye_state = 0  # 0 for diverged, 1 for converged
            self.right_eye_state = 0
            self.prev_left_eye_state = 0
            self.prev_right_eye_state = 0

    def step(self, action, arena, agentlike_objects):
        # if self.energy > self.energy_minimum:
        if True:  # don't let agents die
            if not self.binary_eye_state:
                if self.discrete_actions and not self.use_1dof_eyes:
                    self.turn_left_eye, self.turn_right_eye, discrete_actions = action
                    self.move_forward, self.turn_angle = divmod(
                        discrete_actions, len(self.turn_angle_values)
                    )  # map from set of discrete integers to 2 values
                    self.move_forward = self.move_forward_values[int(self.move_forward)]
                    self.turn_angle = self.turn_angle_values[int(self.turn_angle)]
                elif not self.use_1dof_eyes:
                    (
                        self.turn_left_eye,
                        self.turn_right_eye,
                        self.move_forward,
                        self.turn_angle,
                    ) = action
                    self.move_forward, self.turn_angle = sigmoid(
                        self.move_forward
                    ), np.tanh(self.turn_angle)
                else:
                    self.turn_left_eye, self.move_forward, self.turn_angle = action
                    if not self.eye_muscle_model:  # otherwise we couple the eyes later
                        self.turn_right_eye = (
                            -self.turn_left_eye
                        )  # 1-DOF eyes, turn left and right by same amount
                    self.move_forward, self.turn_angle = sigmoid(
                        self.move_forward
                    ), np.tanh(self.turn_angle)

                if not self.eye_muscle_model:
                    self.turn_left_eye, self.turn_right_eye = np.tanh(
                        self.turn_left_eye
                    ), np.tanh(self.turn_right_eye)
                else:
                    self.turn_left_eye, self.turn_right_eye = sigmoid(
                        self.turn_left_eye
                    ), sigmoid(
                        self.turn_right_eye
                    )  # now represent motor signals instead of turn directions

            else:
                if self.eye_muscle_model:
                    raise ValueError(
                        "Muscle model is not supported with binary eye state."
                    )
                if self.use_eye_fatigue:
                    raise ValueError(
                        "Eye fatigue is not supported with binary eye state."
                    )
                if self.discrete_actions and not self.use_1dof_eyes:
                    left_eye_state, right_eye_state, discrete_actions = action
                    self.move_forward, self.turn_angle = divmod(
                        discrete_actions, len(self.turn_angle_values)
                    )  # map from set of discrete integers to 2 values
                    self.move_forward = self.move_forward_values[int(self.move_forward)]
                    self.turn_angle = self.turn_angle_values[int(self.turn_angle)]
                elif (
                    not self.discrete_actions and not self.use_1dof_eyes
                ):  # continuous actions, 2-DOF eyes, binary eye state
                    self.move_forward, self.turn_angle, left_right_eye_state = action
                    left_eye_state, right_eye_state = divmod(
                        left_right_eye_state, 2
                    )  # map from set of discrete integers to 2 values
                    self.move_forward, self.turn_angle = sigmoid(
                        self.move_forward
                    ), np.tanh(self.turn_angle)
                elif (
                    not self.discrete_actions and self.use_1dof_eyes
                ):  # continuous actions, 1-DOF eyes, binary eye state
                    self.move_forward, self.turn_angle, left_eye_state = action
                    right_eye_state = left_eye_state  # 1-DOF eyes, both eyes are either converged or diverged
                    self.move_forward, self.turn_angle = sigmoid(
                        self.move_forward
                    ), np.tanh(self.turn_angle)
                else:
                    raise NotImplementedError(
                        "Only discrete actions with 2D eyes, continuous actions with 2D eyes, or continuous actions with 1D eyes are supported."
                    )

                self.prev_left_eye_state = self.left_eye_state
                self.prev_right_eye_state = self.right_eye_state
                self.left_eye_state = left_eye_state
                self.right_eye_state = right_eye_state

                if self.eye_wait_time < self.eye_persistence:
                    self.left_eye_state = self.prev_left_eye_state
                    self.right_eye_state = self.prev_right_eye_state
                    self.eye_wait_time += 1

                if self.eye_persistence > 0:
                    if (
                        self.prev_left_eye_state != self.left_eye_state
                        or self.prev_right_eye_state != self.right_eye_state
                    ):
                        self.eye_wait_time = 0

                self.left_eye_angle = (
                    self.min_left_vergence
                    if self.left_eye_state == 0
                    else self.max_left_vergence
                )
                self.right_eye_angle = (
                    self.min_right_vergence
                    if self.right_eye_state == 0
                    else self.max_right_vergence
                )

            self.turn_angle = (
                self.turn_angle * (1 - self.move_forward)
                if self.move_forward > 0
                else 0
            )  # Scale turn angle by 1 - move_forward
            self.move_forward = self.move_forward * (
                1
                + self.np_random.uniform(-self.action_noise_std, self.action_noise_std)
            )  # Uniform noise in range [-action_noise_std, action_noise_std]
            self.turn_angle = self.turn_angle * (
                1
                + self.np_random.uniform(-self.action_noise_std, self.action_noise_std)
            )  # Uniform noise in range [-action_noise_std, action_noise_std]

            if self.fatigue_count > 0:
                self.fatigue_count -= self.fatigue_recovery
                self.fatigue_count = max(0, self.fatigue_count)

            if self.move_forward > 0:
                self.fatigue_count += 1 / ENV_PARAMS["fps_sim"]

            if (
                self.forced_interbout is not None
                and self.interbout_wait_time < self.forced_interbout
            ):
                self.interbout_wait_time += 1
                self.move_forward = 0
                self.turn_angle = 0

            # Calculate new position
            self.orientation += self.max_turn * self.turn_angle
            new_position = self.position.copy()
            if self.move_forward > 0:
                dx = np.cos(self.orientation)
                dy = np.sin(self.orientation)
                new_position += np.array([dx, dy]) * self.move_forward * self.max_step

            self.collided = False

            # Apply perception turn
            if not self.binary_eye_state:
                prev_left_angle = self.left_eye_angle
                prev_right_angle = self.right_eye_angle
                if self.eye_muscle_model:
                    self.turn_left_eye = (
                        -self.k_relax_eye
                        * (self.left_eye_angle - self.min_left_vergence)
                        + self.g_input_eye * self.turn_left_eye
                    )
                    if not self.use_1dof_eyes:
                        self.turn_right_eye = (
                            -self.k_relax_eye
                            * (self.right_eye_angle - self.min_right_vergence)
                            + self.g_input_eye * self.turn_right_eye
                        )
                    else:
                        self.turn_right_eye = -self.turn_left_eye
                    if (
                        (not self.use_eye_fatigue)
                        or self.eye_fatigue_penalty != 0
                        or self.eye_fatigue
                        < 2
                        * (
                            FISH_CONSTANTS["min_right_vergence"]
                            - FISH_CONSTANTS["max_right_vergence"]
                        )
                    ):
                        self.left_eye_angle += self.turn_left_eye
                        self.right_eye_angle += self.turn_right_eye
                else:
                    if (
                        (not self.use_eye_fatigue)
                        or self.eye_fatigue_penalty != 0
                        or self.eye_fatigue
                        < 2
                        * (
                            FISH_CONSTANTS["min_right_vergence"]
                            - FISH_CONSTANTS["max_right_vergence"]
                        )
                    ):
                        self.left_eye_angle += (
                            self.max_perception_turn * self.turn_left_eye
                        )
                        self.right_eye_angle += (
                            self.max_perception_turn * self.turn_right_eye
                        )
                        # print("can move", self.turn_left_eye, self.turn_right_eye, flush=True)

                if self.left_eye_angle < self.min_left_vergence:
                    self.left_eye_angle = self.min_left_vergence
                if self.left_eye_angle > self.max_left_vergence:
                    self.left_eye_angle = self.max_left_vergence
                if self.right_eye_angle > self.min_right_vergence:
                    self.right_eye_angle = self.min_right_vergence
                if self.right_eye_angle < self.max_right_vergence:
                    self.right_eye_angle = self.max_right_vergence

                # Add fatigue based on how much eyes actually moved
                if self.eye_fatigue_penalty != 0 or self.eye_fatigue < 2 * (
                    FISH_CONSTANTS["min_right_vergence"]
                    - FISH_CONSTANTS["max_right_vergence"]
                ):
                    self.eye_fatigue += abs(
                        self.left_eye_angle - prev_left_angle
                    ) + abs(self.right_eye_angle - prev_right_angle)
                if (
                    abs(self.left_eye_angle - prev_left_angle) < 0.01
                    and abs(self.right_eye_angle - prev_right_angle) < 0.01
                ):
                    self.eye_fatigue -= self.eye_fatigue_recovery
                    self.eye_fatigue = max(0, self.eye_fatigue)

            # Check for collisions
            for other_agent in agentlike_objects:
                if other_agent != self:
                    if (
                        np.linalg.norm(new_position - other_agent.position)
                        < 2 * self.body_radius
                    ):
                        # print("Collision between agents", self.agent_id, other_agent.agent_id, flush=True)
                        self.collided = True  # Track collision state
                        # self.move_forward = 0
                        # self.turn_angle = 0
                        break

            # If no collisions, update position
            if not self.collided:
                self.position = new_position

            self.check_and_consume_food(arena)

            # Update energy levels
            self.energy -= self.energy_per_step * (1 + abs(self.move_forward))
            self.energy = np.clip(self.energy, self.energy_minimum, self.energy_maximum)

            # Update trajectory history
            self.trajectory.append(self.position.copy())
            self.trajectory = self.trajectory[-5:]

            self.last_action = [
                self.move_forward,
                self.turn_angle,
                self.turn_left_eye,
                self.turn_right_eye,
            ]

            if self.forced_interbout is not None and (
                self.move_forward > 0 or abs(self.turn_angle) > 0
            ):
                self.interbout_wait_time = 1

            # Handle arena boundaries based on arena shape
            # Handle circular arena boundary collision
            center = arena.center
            distance_from_center = np.linalg.norm(self.position - center)

            buffer = self.body_radius + self.eye_hypotenuse
            if distance_from_center > arena.arena_radius - buffer:
                # Clip to circular boundary
                direction = (self.position - center) / distance_from_center
                self.position = center + direction * (
                    arena.arena_radius - buffer
                )
            
    def check_and_consume_food(self, arena):
        """
        Check if the agent is within eating radius / eating angle of any food, and consume it one by one starting from the closest.
        """
        self.curr_food_consumed = []
        if arena.food_positions.size == 0:
            return False

        # Get distances to all food positions
        distances = cdist([self.position], arena.food_positions)[0]
        within_eating_radius = distances < self.eating_radius
        indices = np.where(within_eating_radius)[0]

        eatable_food = []
        for i, idx in enumerate(indices):
            food_pos = arena.food_positions[idx]
            food_id = arena.food_pellets[idx].global_index
            distance_to_food = distances[idx]

            # Calculate angle from agent orientation to food
            food_direction = food_pos - self.position
            food_angle = np.arctan2(food_direction[1], food_direction[0])

            # Calculate angle difference from agent's forward direction
            angle_diff = food_angle - self.orientation

            # Normalize angle difference to [-π, π]
            angle_diff = np.arctan2(np.sin(angle_diff), np.cos(angle_diff))

            # Check if food is within eating angle cone
            if abs(angle_diff) <= self.eating_angle / 2:
                eating_prob = np.exp(-np.abs(angle_diff * self.eating_distribution_decay))  # Decay based on angle
                eating_prob /= np.exp(-self.baseline_success_eating_angle * self.eating_distribution_decay)  # Normalize by baseline
                if self.np_random.uniform(0, 1) < eating_prob:
                    eatable_food.append((idx, distance_to_food, food_id))

        if len(eatable_food) > 0:
            # Sort by distance (closest first)
            eatable_food.sort(key=lambda x: x[1])

            # Determine how many foods can be eaten this step
            max_food_to_eat = (
                min(self.max_food_eaten_per_step, len(eatable_food))
                if self.max_food_eaten_per_step is not None
                else len(eatable_food)
            )

            # Eat the closest foods within the eating cone
            for idx, _, food_id in eatable_food[:max_food_to_eat]:
                arena.eat_food(idx)
                self.food_consumed_ids.append(food_id)
                self.curr_food_consumed.append(idx)
                self.energy += self.energy_food
            return True

        return False

        # self.curr_food_consumed = []
        # if arena.food_positions.size == 0:
        #     return False

        # # Get distances to all food positions
        # distances = cdist([self.position], arena.food_positions)[0]
        # within_eating_radius = distances < self.eating_radius
        # indices = np.where(within_eating_radius)[0]

        # if len(indices) > 0:
        #     # Sort indices by distance to the agent
        #     sorted_indices = indices[np.argsort(distances[indices])]
        #     # Determine how many foods can be eaten this step
        #     max_food_to_eat = min(self.max_food_eaten_per_step, len(sorted_indices)) if self.max_food_eaten_per_step is not None else len(sorted_indices)
        #     # Eat the closest foods, one by one
        #     for idx in sorted_indices[:max_food_to_eat]:
        #         arena.eat_food(idx)  # Eat food one by one using the existing method
        #         self.curr_food_consumed.append(idx)
        #         self.energy += self.energy_food
        #     return True
        # return False

    def _get_eye_positions(self):
        """Calculate the world positions of left and right eyes based on agent position and orientation."""
        # Calculate eye positions relative to agent center
        # Eyes are positioned forward and to the sides of the agent

        # Forward direction
        forward_x = np.cos(self.orientation) * self.eye_forward_offset
        forward_y = np.sin(self.orientation) * self.eye_forward_offset

        # Side direction (perpendicular to forward)
        side_x = -np.sin(self.orientation) * self.eye_separation / 2
        side_y = np.cos(self.orientation) * self.eye_separation / 2

        # Left eye position (to the left when looking forward)
        left_eye_pos = self.position + np.array(
            [forward_x - side_x, forward_y - side_y]
        )

        # Right eye position (to the right when looking forward)
        right_eye_pos = self.position + np.array(
            [forward_x + side_x, forward_y + side_y]
        )

        return left_eye_pos, right_eye_pos

    def _update_eye_sensors(
        self, eye_pos, eye_angle, sensors, agent_positions, food_positions, food_pellets
    ):
        detected_objects = (
            {}
        )  # Track objects detected by this eye: {object_id: (distance, type, position)}

        # Agent Detection
        if agent_positions:
            agent_positions = np.array(agent_positions)
            relative_positions = agent_positions - eye_pos
            agent_distances = np.linalg.norm(relative_positions, axis=1)
            # print(f"Found {len(agent_positions)} agents {agent_distances}", flush=True)

            in_range_mask = (agent_distances <= self.fish_sensing_radius) & (
                agent_distances > 1e-6
            )  # get other agents in sensing range

            if np.any(in_range_mask):
                relative_positions = relative_positions[in_range_mask]
                agent_distances = agent_distances[in_range_mask]
                angles = (
                    np.arctan2(relative_positions[:, 1], relative_positions[:, 0])
                    - self.orientation
                )

                angles = (angles + np.pi) % (2 * np.pi) - np.pi

                # Add noise to angles
                if not self.conspecific_monocular_perception:
                    angles += self.np_random.uniform(
                        -self.angle_noise_std_walker,
                        self.angle_noise_std_walker,
                        size=angles.shape
                    )

                sorted_indices = np.argsort(-agent_distances)
                sorted_distances = agent_distances[sorted_indices]
                sorted_angles = angles[sorted_indices]
                sorted_positions = agent_positions[in_range_mask][sorted_indices]

                for distance, angle, position in zip(
                    sorted_distances, sorted_angles, sorted_positions
                ):  # sort by distance (furthest to closest)
                    if (
                        angle > eye_angle - self.perception_field / 2
                        and angle < eye_angle + self.perception_field / 2
                    ):
                        noisy_distance = distance * (
                            1
                            + self.np_random.uniform(
                                low=-self.distance_noise_std,
                                high=self.distance_noise_std,
                            )
                        )
                        # clip noisy distance to be within 0 and sensing radius
                        noisy_distance = np.clip(noisy_distance, 0, self.fish_sensing_radius)
                        normalized_distance = 1 - (noisy_distance / self.ray_length)
                        sensor_index = int(
                            (angle - (eye_angle - self.perception_field / 2))
                            // self.projected_sector_angle
                        )

                        # Create unique object ID for agents (use position as identifier)
                        object_id = f"agent_{tuple(np.round(position, 2))}"
                        detected_objects[object_id] = (
                            normalized_distance,
                            OBJECT_TYPES["AGENT"],
                            position,
                            sensor_index,
                        )

                        if normalized_distance > sensors[sensor_index][0]:
                            sensors[sensor_index] = [
                                normalized_distance,
                                OBJECT_TYPES["AGENT"],
                            ]
                            # print("Sensor ", sensor_index, " sees agent at distance ", normalized_distance, flush=True)

                    self.sensed_agents.append(position)

        # Food Detection
        if food_positions.size > 0:
            food_positions = np.array(food_positions)
            relative_positions = food_positions - eye_pos
            food_distances = np.linalg.norm(relative_positions, axis=1)

            # Filter out food that are out of projected sensing range
            in_range_mask = food_distances <= self.food_sensing_radius
            if np.any(in_range_mask):
                filtered_indices = np.where(in_range_mask)[0]
                relative_positions = relative_positions[in_range_mask]
                food_distances = food_distances[in_range_mask]
                angles = (
                    np.arctan2(relative_positions[:, 1], relative_positions[:, 0])
                    - self.orientation
                )

                # Normalize angles to the range [-π, π]
                angles = (angles + np.pi) % (2 * np.pi) - np.pi

                # Add noise to angles
                angles += self.np_random.uniform(
                    -self.angle_noise_std_food,
                    self.angle_noise_std_food,
                    size=angles.shape
                )

                # Sort by distance (furthest to closest)
                sorted_indices = np.argsort(-food_distances)
                sorted_distances = food_distances[sorted_indices]
                sorted_angles = angles[sorted_indices]
                sorted_positions = food_positions[in_range_mask][sorted_indices]

                for rel_idx, distance, angle in zip(
                    sorted_indices, sorted_distances, sorted_angles
                ):
                    pellet_idx = filtered_indices[
                        rel_idx
                    ]  # Get the original index of the food pellet
                    pellet_id = food_pellets[pellet_idx].global_index

                    if (
                        angle > eye_angle - self.perception_field / 2
                        and angle < eye_angle + self.perception_field / 2
                    ):
                        noisy_distance = distance * (
                            1
                            + self.np_random.uniform(
                                low=-self.distance_noise_std,
                                high=self.distance_noise_std,
                            )
                        )  # Uniform noise in range [-distance_noise_std, distance_noise_std]
                        # clip noisy distance to be within 0 and sensing radius
                        noisy_distance = np.clip(noisy_distance, 0, self.food_sensing_radius)

                        normalized_distance = 1 - (
                            noisy_distance / self.ray_length
                        )  # Note: always use same ray length for normalization
                        sensor_index = int(
                            (angle - (eye_angle - self.perception_field / 2))
                            // self.projected_sector_angle
                        )

                        # Use pellet ID as object identifier
                        object_id = f"food_{pellet_id}"
                        detected_objects[object_id] = (
                            normalized_distance,
                            OBJECT_TYPES["FOOD"],
                            sorted_positions[rel_idx],
                            sensor_index,
                        )

                        # Set the sensor value only if the new reading is stronger (closer object)
                        if normalized_distance > sensors[sensor_index][0]:
                            sensors[sensor_index] = [
                                normalized_distance,
                                OBJECT_TYPES["FOOD"],
                            ]
                            # print("Sensor ", sensor_index, " sees food at distance ", normalized_distance, flush=True)
                            self.detected_food_ids.append(pellet_id)

        # Wall Detection

        # For sensors that have zero value (no agent or food detected), check for walls
        for sensor_index in range(self.num_sensors):
            if sensors[sensor_index][0] == 0:
                # Compute the center angle of this sensor
                sector_center_angle = (
                    eye_angle
                    - self.perception_field / 2
                    + (sensor_index + 0.5) * self.projected_sector_angle
                )
                global_angle = sector_center_angle + self.orientation

                # Circular arena wall detection
                arena_radius = self.arena_radius
                center = np.array([arena_radius, arena_radius])
                ray_direction = np.array(
                    [np.cos(global_angle), np.sin(global_angle)]
                )

                # Ray-circle intersection: solve |eye_pos + t*ray_direction - center|² = radius²
                to_eye = eye_pos - center
                a = 1.0  # ray_direction is unit vector, so dot(ray_direction, ray_direction) = 1
                b = 2.0 * np.dot(to_eye, ray_direction)
                c = np.dot(to_eye, to_eye) - arena_radius**2

                discriminant = b**2 - 4 * a * c
                if discriminant >= 0:
                    t1 = (-b + np.sqrt(discriminant)) / (2 * a)
                    t2 = (-b - np.sqrt(discriminant)) / (2 * a)

                    # Take the closest positive intersection (forward direction)
                    valid_t = [
                        t for t in [t1, t2] if t > 1e-6
                    ]  # Small epsilon to avoid self-intersection
                    if valid_t:
                        min_distance = min(valid_t)

                        if min_distance <= self.wall_sensing_radius:
                            # Add noise
                            noisy_distance = min_distance * (
                                1
                                + self.np_random.uniform(
                                    low=-self.distance_noise_std,
                                    high=self.distance_noise_std,
                                )
                            )
                            # Stay within bounds 
                            noisy_distance = np.clip(noisy_distance, 0, self.wall_sensing_radius)

                            normalized_distance = 1 - (
                                noisy_distance / self.ray_length
                            )
                            sensors[sensor_index] = [
                                normalized_distance,
                                OBJECT_TYPES["WALL"],
                            ]

                            # Store detection info
                            object_id = f"wall_{sensor_index}"
                            wall_position = (
                                eye_pos + min_distance * ray_direction
                            )
                            detected_objects[object_id] = (
                                normalized_distance,
                                OBJECT_TYPES["WALL"],
                                wall_position,
                                sensor_index,
                            )
            

        # Each sensor will fail with probability self.detection_failure_rate
        for sensor_index in range(self.num_sensors):
            if self.np_random.random() < self.detection_failure_rate:
                sensors[sensor_index] = [0, OBJECT_TYPES["NONE"]]

        # Each sensor detects false positive with probability self.false_positive_rate
        for sensor_index in range(self.num_sensors):
            if self.np_random.random() < self.false_positive_rate:
                sensors[sensor_index] = [self.np_random.uniform(), OBJECT_TYPES["FOOD"]]

        # Check if eyes are outside arena boundaries
        arena_center = np.array([self.arena_radius, self.arena_radius])
        eye_distance_from_center = np.linalg.norm(eye_pos - arena_center)
        if eye_distance_from_center > self.arena_radius:
            # If eye is outside circular arena, all sensors detect wall
            for sensor_index in range(self.num_sensors):
                sensors[sensor_index] = [1, OBJECT_TYPES["WALL"]]
        
        if False: # debug print
            for sensor_index in range(self.num_sensors):
                if sensors[sensor_index][1] == OBJECT_TYPES["AGENT"]:
                    print("Sensor ", sensor_index, "STILL sees agent at distance ", sensors[sensor_index][0], flush=True)
                if sensors[sensor_index][1] == OBJECT_TYPES["FOOD"]:
                    print("Sensor ", sensor_index, "STILL sees food at distance ", sensors[sensor_index][0], flush=True)

        return detected_objects

    def update_projected_sensors(self, agent_positions, food_positions, food_pellets):
        # NOTE: binocular depth only is only supported with separate eyes
        self.sensed_agents = []  # Reset the list each step
        self.detected_food_ids = []  # Reset the list each step
        self.left_eye_sensors = np.zeros(
            (self.num_sensors, 2)
        )  # Now storing [normalized_distance, object_type]
        self.right_eye_sensors = np.zeros(
            (self.num_sensors, 2)
        )  # Now storing [normalized_distance, object_type]
        left_eye_pos, right_eye_pos = self._get_eye_positions()

        if self.flash_monocular_only:
            self.prev_left_detections = (
                self.left_detections.copy()
            )  # Store previous left eye detections
            self.prev_right_detections = (
                self.right_detections.copy()
            )  # Store previous right eye detections

        # Get detections from both eyes
        self.left_detections = self._update_eye_sensors(
            left_eye_pos,
            self.left_eye_angle,
            self.left_eye_sensors,
            agent_positions,
            food_positions,
            food_pellets,
        )
        self.right_detections = self._update_eye_sensors(
            right_eye_pos,
            self.right_eye_angle,
            self.right_eye_sensors,
            agent_positions,
            food_positions,
            food_pellets,
        )

        # print("left: ", self.left_eye_sensors, flush=True)
        # print("right: ", self.right_eye_sensors, flush=True)

        # Apply binocular depth processing
        if self.binocular_depth_only:
            self._apply_binocular_depth_processing()

        if self.binocular_angle_only:
            self._apply_binocular_angle_only_processing()

        if self.flash_monocular_only:
            self._apply_flash_monocular_processing()

    def _apply_flash_monocular_processing(self):
        """
        Apply flash monocular processing: if an object is detected by one eye but not the other,
        and this object was also detected in the previous step, remove it from the sensors.
        """
        if not hasattr(self, "prev_left_detections") or not hasattr(
            self, "prev_right_detections"
        ):
            return

        # Find objects detected by left eye but not right eye
        left_only_objects = set(self.left_detections.keys()) - set(
            self.right_detections.keys()
        )

        # Find objects detected by right eye but not left eye
        right_only_objects = set(self.right_detections.keys()) - set(
            self.left_detections.keys()
        )

        if self.conspecific_monocular_perception:
            # Exclude agents from monocular removal
            left_only_objects = {
                obj_id
                for obj_id in left_only_objects
                if self.left_detections[obj_id][1] != OBJECT_TYPES["AGENT"]
            }
            right_only_objects = {
                obj_id
                for obj_id in right_only_objects
                if self.right_detections[obj_id][1] != OBJECT_TYPES["AGENT"]
            }

        # Process left-only objects
        for obj_id in left_only_objects:
            # Check if this object was also detected by left eye in previous step
            if obj_id in self.prev_left_detections:
                # Get sensor index and remove detection
                _, _, _, sensor_idx = self.left_detections[obj_id]
                self.left_eye_sensors[sensor_idx] = [0, OBJECT_TYPES["NONE"]]

        # Process right-only objects
        for obj_id in right_only_objects:
            # Check if this object was also detected by right eye in previous step
            if obj_id in self.prev_right_detections:
                # Get sensor index and remove detection
                _, _, _, sensor_idx = self.right_detections[obj_id]
                self.right_eye_sensors[sensor_idx] = [0, OBJECT_TYPES["NONE"]]

    def _apply_binocular_depth_processing(self):
        """
        Apply binocular depth processing: only provide distance information when both eyes
        detect the same object, otherwise provide only object type information.
        Exception: walls always keep distance information (and agents if conspecific_monocular_perception == True).
        """
        left_detections = self.left_detections
        right_detections = self.right_detections
        common_objects = set(left_detections.keys()) & set(right_detections.keys())

        if not common_objects:
            # Clear all distances at once
            if not self.conspecific_monocular_perception:
                left_mask = (self.left_eye_sensors[:, 0] > 0) & (
                    self.left_eye_sensors[:, 1] != OBJECT_TYPES["WALL"]
                )
                right_mask = (self.right_eye_sensors[:, 0] > 0) & (
                    self.right_eye_sensors[:, 1] != OBJECT_TYPES["WALL"]
                )
            else:
                left_mask = (
                    (self.left_eye_sensors[:, 0] > 0)
                    & (self.left_eye_sensors[:, 1] != OBJECT_TYPES["WALL"])
                    & (self.left_eye_sensors[:, 1] != OBJECT_TYPES["AGENT"])
                )
                right_mask = (
                    (self.right_eye_sensors[:, 0] > 0)
                    & (self.right_eye_sensors[:, 1] != OBJECT_TYPES["WALL"])
                    & (self.right_eye_sensors[:, 1] != OBJECT_TYPES["AGENT"])
                )
            self.left_eye_sensors[left_mask, 0] = UNKNOWN_DIST
            self.right_eye_sensors[right_mask, 0] = UNKNOWN_DIST

        # Create global set of all objects seen by each eye
        all_left_objects = set()
        all_right_objects = set()

        # Maps to track which sensors see each common object
        left_obj_to_sensors = {}
        right_obj_to_sensors = {}

        for obj_id in common_objects:
            if obj_id in left_detections:
                _, _, _, sensor_idx = left_detections[obj_id]
                all_left_objects.add(obj_id)
                if obj_id not in left_obj_to_sensors:
                    left_obj_to_sensors[obj_id] = set()
                left_obj_to_sensors[obj_id].add(sensor_idx)

            if obj_id in right_detections:
                _, _, _, sensor_idx = right_detections[obj_id]
                all_right_objects.add(obj_id)
                if obj_id not in right_obj_to_sensors:
                    right_obj_to_sensors[obj_id] = set()
                right_obj_to_sensors[obj_id].add(sensor_idx)

        # Find sensors that see binocular objects
        left_binocular_sensors = set()
        right_binocular_sensors = set()

        for obj_id in common_objects:
            if obj_id in left_obj_to_sensors and obj_id in right_obj_to_sensors:
                left_binocular_sensors.update(left_obj_to_sensors[obj_id])
                right_binocular_sensors.update(right_obj_to_sensors[obj_id])

        # Create boolean masks for vectorized operations
        all_sensors = set(range(self.num_sensors))
        left_remove_sensors = all_sensors - left_binocular_sensors
        right_remove_sensors = all_sensors - right_binocular_sensors

        # Vectorized distance removal
        if left_remove_sensors:
            left_indices = list(left_remove_sensors)
            if not self.conspecific_monocular_perception:
                active_mask = (self.left_eye_sensors[left_indices, 0] > 0) & (
                    self.left_eye_sensors[left_indices, 1] != OBJECT_TYPES["WALL"]
                )
            else:
                active_mask = (
                    (self.left_eye_sensors[left_indices, 0] > 0)
                    & (self.left_eye_sensors[left_indices, 1] != OBJECT_TYPES["WALL"])
                    & (self.left_eye_sensors[left_indices, 1] != OBJECT_TYPES["AGENT"])
                )
            active_indices = np.array(left_indices)[active_mask]
            if len(active_indices) > 0:
                self.left_eye_sensors[active_indices, 0] = UNKNOWN_DIST

        if right_remove_sensors:
            right_indices = list(right_remove_sensors)
            if not self.conspecific_monocular_perception:
                active_mask = (self.right_eye_sensors[right_indices, 0] > 0) & (
                    self.right_eye_sensors[right_indices, 1] != OBJECT_TYPES["WALL"]
                )
            else:
                active_mask = (
                    (self.right_eye_sensors[right_indices, 0] > 0)
                    & (self.right_eye_sensors[right_indices, 1] != OBJECT_TYPES["WALL"])
                    & (
                        self.right_eye_sensors[right_indices, 1]
                        != OBJECT_TYPES["AGENT"]
                    )
                )
            active_indices = np.array(right_indices)[active_mask]
            if len(active_indices) > 0:
                self.right_eye_sensors[active_indices, 0] = UNKNOWN_DIST

    def _apply_binocular_angle_only_processing(self):
        """
        Apply binocular angle-only processing: randomly reassign UNKNOWN_DIST signals to different sensors.
        """
        # For binocular angle only, randomly reassign UNKNOWN_DIST signals to different sensors
        left_unknown_indices = np.where((self.left_eye_sensors[:, 0] == UNKNOWN_DIST))[
            0
        ]
        right_unknown_indices = np.where(
            (self.right_eye_sensors[:, 0] == UNKNOWN_DIST)
        )[0]

        # Randomly reassign left eye UNKNOWN_DIST signals
        if len(left_unknown_indices) > 0:
            # Get all sensors that currently have no detection
            # left_nonactive_indices = np.where(self.left_eye_sensors[:, 1] == OBJECT_TYPES["NONE"])[0]
            if len(self.left_eye_sensors) >= len(left_unknown_indices):
                # Randomly select which non-active sensors to replace with UNKNOWN_DIST
                new_unknown_indices = self.np_random.choice(
                    len(self.left_eye_sensors),
                    size=len(left_unknown_indices),
                    replace=False,
                )
                new_unknown_indices = []
                for old_idx in left_unknown_indices:
                    # Move to opposite end of sensor array
                    if old_idx < len(self.left_eye_sensors) // 2:
                        new_idx = len(self.left_eye_sensors) - 1 - old_idx
                    else:
                        new_idx = len(self.left_eye_sensors) - 1 - old_idx
                    new_unknown_indices.append(new_idx)
                # For each unknown index, get its object type and reassign to a new sensor
                for old_idx, new_idx in zip(left_unknown_indices, new_unknown_indices):
                    object_type = self.left_eye_sensors[old_idx, 1]
                    # Clear original UNKNOWN_DIST sensor
                    self.left_eye_sensors[old_idx, 0] = 0
                    self.left_eye_sensors[old_idx, 1] = OBJECT_TYPES["NONE"]
                    # Set new sensor to UNKNOWN_DIST with the original object type
                    self.left_eye_sensors[new_idx, 0] = UNKNOWN_DIST
                    self.left_eye_sensors[new_idx, 1] = object_type

        # Randomly reassign right eye UNKNOWN_DIST signals
        if len(right_unknown_indices) > 0:
            # Get all sensors that currently have no detection
            # right_nonactive_indices = np.where(self.right_eye_sensors[:, 1] == OBJECT_TYPES["NONE"])[0]
            if len(self.right_eye_sensors) >= len(right_unknown_indices):
                # Randomly select which non-active sensors to replace with UNKNOWN_DIST
                # new_unknown_indices = self.np_random.choice(len(self.right_eye_sensors),
                #                                           size=len(right_unknown_indices),
                #                                           replace=False)
                new_unknown_indices = []
                for old_idx in right_unknown_indices:
                    # Move to opposite end of sensor array
                    if old_idx < len(self.right_eye_sensors) // 2:
                        new_idx = len(self.right_eye_sensors) - 1 - old_idx
                    else:
                        new_idx = len(self.right_eye_sensors) - 1 - old_idx
                    new_unknown_indices.append(new_idx)
                # For each unknown index, get its object type and reassign to a new sensor
                for old_idx, new_idx in zip(right_unknown_indices, new_unknown_indices):
                    object_type = self.right_eye_sensors[old_idx, 1]
                    # Clear original UNKNOWN_DIST sensor
                    self.right_eye_sensors[old_idx, 0] = 0
                    self.right_eye_sensors[old_idx, 1] = OBJECT_TYPES["NONE"]
                    # Set new sensor to UNKNOWN_DIST with the original object type
                    self.right_eye_sensors[new_idx, 0] = UNKNOWN_DIST
                    self.right_eye_sensors[new_idx, 1] = object_type

    def _line_segment_intersection(self, p1, p2, q1, q2):
        # Check if two line segments (p1, p2) and (q1, q2) intersect
        # Returns the distance from p1 to the intersection point, or None if no intersection
        r = p2 - p1
        s = q2 - q1
        det = r[0] * s[1] - r[1] * s[0]
        if det == 0:
            return None  # Lines are parallel
        t = -((q1[1] - p1[1]) * s[0] - (q1[0] - p1[0]) * s[1]) / det
        u = -((q1[1] - p1[1]) * r[0] - (q1[0] - p1[0]) * r[1]) / det
        if 0 <= t <= 1 and 0 <= u <= 1:
            intersection_point = p1 + t * r
            return np.linalg.norm(intersection_point - p1)
        return None


class WalkerBot:
    """
    A simple, non-interactive walker that does a random walk.
    - Randomly initialized within the arena.
    - Each step: random small turn (bounded by max_turn) and forward step (bounded by max_step).
    - Kept inside arena bounds.
    """

    def __init__(
        self,
        arena_size,
        max_step=1.0,
        max_turn=0.3,
        seed=None,
        center_seeking_alpha=10,
        pursuit=True,
        agent_perception_radius=10,  # for pursuit behavior (in mm)
        ignore_walls=False,
    ):
        self.arena_size = np.array(arena_size, dtype=float)
        self.max_step = float(max_step)
        self.max_turn = float(max_turn)
        self.body_radius = AGENT_PARAMS.get("body_radius", 1.0)
        self.np_random = np.random.default_rng(seed)
        self.position = np.zeros(2, dtype=float)
        self.orientation = 0.0
        self.trajectory = []
        # set later for circle arenas
        self.arena_radius = None
        self.arena_center = None
        self.center_seeking_alpha = center_seeking_alpha  # exponent for downweighting center-seeking probability (higher --> less center-seeking)
        self.pursuit = pursuit  # if true, be attracted to agents
        self.agent_perception_radius = agent_perception_radius
        self.ignore_walls = ignore_walls  # if true, ignore walls when pursuing agents

    def reset(self, seed=None):
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        # random position/orientation
        angle = self.np_random.uniform(0, 2 * np.pi)
        radius = np.sqrt(self.np_random.uniform(0, 1)) * (
            self.arena_radius - self.body_radius
        )
        center = np.array([self.arena_radius, self.arena_radius])
        self.position = center + radius * np.array([np.cos(angle), np.sin(angle)])
        self.orientation = self.np_random.uniform(-np.pi, np.pi)
        self.trajectory = [self.position.copy()]
        self.override = False
        self.pursuing = False

    def step(self, agent_positions=None):
        """
        Random walk with center-seeking near boundaries.
        - Proposes a random step (rand_forward, rand_turn).
        - If the candidate leaves bounds or with probability (1 - normalized_distance_to_center),
          turn toward the arena center instead, and scale forward by that probability.
        """
        # propose a random step
        rand_turn = self.np_random.uniform(-self.max_turn, self.max_turn)
        rand_forward = self.np_random.uniform(0.0, self.max_step)
        cand_turn = rand_turn
        cand_forward = rand_forward
        cand_theta = normalize_angle(self.orientation + rand_turn)
        cand_delta = np.array([np.cos(cand_theta), np.sin(cand_theta)]) * rand_forward
        self.pursuing = False
        if self.pursuit and agent_positions is not None:
            agent_positions = np.array(agent_positions)
            relative_positions = agent_positions - self.position
            distances = np.linalg.norm(relative_positions, axis=1)
            in_range_mask = (distances <= self.agent_perception_radius) & (distances > 1e-6)
            if np.any(in_range_mask):
                relative_positions = relative_positions[in_range_mask]
                distances = distances[in_range_mask]
                closest_idx = np.argmin(distances)
                target_direction = relative_positions[closest_idx]
                desired_heading = np.arctan2(target_direction[1], target_direction[0])
                error = normalize_angle(desired_heading - self.orientation)
                cand_turn = np.sign(error) * min(self.max_turn, abs(error))
                cand_forward = self.max_step
                cand_theta = normalize_angle(self.orientation + cand_turn)
                cand_delta = np.array([np.cos(cand_theta), np.sin(cand_theta)]) * cand_forward
                self.pursuing = True

        cand_pos = self.position + cand_delta

        # geometry helpers
        center = (
            self.arena_center
            if self.arena_center is not None
            else np.array([self.arena_radius, self.arena_radius])
        )
        bounding_radius = self.arena_radius - self.body_radius
        dist_to_center = np.linalg.norm(cand_pos - center)
        within_bounds = dist_to_center <= bounding_radius
        normalized_distance = np.clip(
            dist_to_center / max(bounding_radius, 1e-9), 0.0, 1.0
        )

        # probability to keep random walk decreases near edge
        p_random = 1.0 - normalized_distance**self.center_seeking_alpha
        p_random = float(np.clip(p_random, 0.0, 1.0))
        do_walk = self.np_random.random() < p_random

        if self.ignore_walls:
            # always move, as long as within bounds
            do_walk = True

        if within_bounds and do_walk:
            # accept random (or targeted) step
            turn = cand_turn
            forward = cand_forward
            self.override = False
        else:
            # face back toward center
            self.override = True
            vec_to_center = center - self.position
            desired_heading = np.arctan2(vec_to_center[1], vec_to_center[0])
            # heading error in [-pi, pi]
            error = normalize_angle(desired_heading - self.orientation)
            # strong but bounded turn toward the center
            turn = np.sign(error) * self.max_turn
            # smaller forward near edge; scale by p_random
            forward = self.max_step * self.np_random.uniform(0.0, p_random)

        # apply chosen motion
        self.orientation = normalize_angle(self.orientation + turn)
        delta = np.array([np.cos(self.orientation), np.sin(self.orientation)]) * forward
        new_pos = self.position + delta

        # final clamp to arena
        center_c = (
            self.arena_center
            if self.arena_center is not None
            else np.array([self.arena_radius, self.arena_radius])
        )
        vec = new_pos - center_c
        d = np.linalg.norm(vec)
        max_r = self.arena_radius - self.body_radius
        if d > max_r:
            vec = vec / (d + 1e-9) * max_r
            new_pos = center_c + vec

        self.position = new_pos
        self.trajectory.append(self.position.copy())
        self.trajectory = self.trajectory[-15:]


class MultiAgentZFishEnv(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array", None],
        "name": "MultiAgentZFish_v1",
    }

    def __init__(
        self,
        all_args,
        reset_callback=None,
        reward_callback=None,
        observation_callback=None,
        info_callback=None,
        done_callback=None,
        post_step_callback=None,
        is_eval=False,
        seed=None,
    ):
        self.all_args = all_args
        if self.all_args.get("cfg_override", None) is not None:
            FISH_CONSTANTS.update(self.all_args["cfg_override"]["FISH_CONSTANTS"])
            ENV_PARAMS.update(self.all_args["cfg_override"]["ENV_PARAMS"])
            AGENT_PARAMS.update(self.all_args["cfg_override"]["AGENT_PARAMS"])
            REWARDS.update(self.all_args["cfg_override"]["REWARDS"])
            OBJECT_TYPES.update(self.all_args["cfg_override"]["OBJECT_TYPES"])

        print("Updated rewards:", REWARDS["eat"], REWARDS["collision"])

        self.__dict__.update(ENV_PARAMS)
        self.current_episode = 0
        self.use_food_decay = all_args.get("use_food_decay", False)
        self.num_agents = all_args["num_agents"]
        self.energy_minimum = AGENT_PARAMS["energy_minimum"]
        self.arena_size = all_args.get(
            "arena_size", (200, 200)
        )  # Default is temporary for initialization
        self.max_step = ENV_PARAMS["max_step"]
        self.max_turn = ENV_PARAMS["max_turn"]
        self.max_perception_turn = FISH_CONSTANTS["max_eye_turn"]
        self.distance_noise_std = AGENT_PARAMS[
            "distance_noise_std"
        ]  # Standard deviation of noise added to distance measurement
        self.detection_failure_rate = all_args[
            "detection_failure_rate"
        ]  # Probability of complete detection failure
        self.false_positive_rate = all_args[
            "false_positive_rate"
        ]  # Probability of false positive detection
        self.shared_reward = all_args.get("shared_reward", False)
        self.expt_name = None
        self.timestamp = all_args["timestamp"]
        self.video_save_dir = (
            f"{all_args['run_dir']}/outputs" if "run_dir" in all_args else "./"
        )
        self.is_eval = is_eval
        self.eating_distribution_decay = all_args["eating_distribution_decay_start"]
        self.eating_distribution_decay_start = all_args["eating_distribution_decay_start"]
        self.eating_distribution_decay_max = all_args["eating_distribution_decay_max"]
        self.eating_distribution_decay_step = all_args.get("eating_distribution_decay_step", 0.0)
        self.food_speed = all_args.get("food_speed_start", 0.0)
        self.food_speed_max = all_args["food_speed_max"]
        self.food_speed_increase = all_args["food_speed_step"]
        self.food_turn_std = all_args.get("food_turn_std_start", 0.0)
        self.food_turn_std_max = all_args["food_turn_std_max"]
        self.food_turn_std_increase = all_args["food_turn_std_step"]
        self.train_food_scaling_min = all_args.get("train_food_scaling_min", 1.0)
        self.train_food_scaling_max = all_args.get("train_food_scaling_max", 1.0)
        self.train_food_scaling_type = all_args.get(
            "train_food_scaling_type", "uniform"
        )
        self.min_food_density = all_args.get("min_food_density", ENV_PARAMS["min_food_density"])

        self.r_collide_start = all_args.get("r_collide_start")
        self.r_collide_end = all_args.get("r_collide_end")
        self.r_collide_step = all_args.get("r_collide_step")

        if self.r_collide_start is None:
            self.r_collide_start = REWARDS["collision"]

        if self.r_collide_end is None:
            self.r_collide_end = REWARDS["collision"]
        
        if self.r_collide_step is None:
            self.r_collide_step = 0.0


        self.total_num_episodes = all_args["num_env_steps"] // all_args["max_episode_length"] // all_args["n_rollout_threads"]

        self.agent_objects = [
            ZFishAgent(
                all_args,
                self.arena_size,
                self.max_step,
                self.max_turn,
                self.max_perception_turn,
                agent_id,
                self.energy_minimum,
                self.eating_distribution_decay,
            )
            for agent_id in range(self.num_agents)
        ]

        self.num_walkerbots = int(all_args.get("num_walkerbots", 0))
        self.walker_max_step = ENV_PARAMS["walker_max_step"]
        self.walker_max_turn = ENV_PARAMS["walker_max_turn"]
        self.walker_agent_perception_radius = ENV_PARAMS["walker_agent_perception_radius"]
        self.walker_pursuit = all_args.get("walker_pursuit", False)
        self.walker_ignore_walls = all_args.get("walker_ignore_walls", False)

        self.arena_mapping = {
            ar.UniformArena.__name__: ar.UniformArena,
            ar.PatchyArena.__name__: ar.PatchyArena,
            ar.FeederArena.__name__: ar.FeederArena,
        }
        sum_proportions = (
            self.all_args["pfeeder"]
            + self.all_args["prandom"]
            + self.all_args["urandom"]
        )
        self.arena_generators = [
            (ar.FeederArena.__name__, self.all_args["pfeeder"] / sum_proportions),
            (ar.PatchyArena.__name__, self.all_args["prandom"] / sum_proportions),
            (ar.UniformArena.__name__, self.all_args["urandom"] / sum_proportions),
        ]

        self.fig, self.ax = None, None
        self.frames = []
        self.render_mode = all_args.get("render_mode", None)

        self.reset_callback = reset_callback
        self.reward_callback = reward_callback
        self.observation_callback = observation_callback
        self.info_callback = info_callback
        self.done_callback = done_callback
        self.post_step_callback = post_step_callback

        self.action_space = []
        self.observation_space = []
        self.move_forward_values = self.agent_objects[
            0
        ].move_forward_values  # Use the first agent's values
        self.turn_values = self.agent_objects[
            0
        ].turn_angle_values  # Use the first agent's values

        self.penalize_move_threshold = AGENT_PARAMS.get("penalize_move_threshold", None)
        self.penalize_turn_threshold = AGENT_PARAMS.get("penalize_turn_threshold", None)

        for agent in range(self.num_agents):
            if not self.agent_objects[agent].binary_eye_state:
                if (
                    self.agent_objects[agent].discrete_actions
                    and not self.agent_objects[agent].use_1dof_eyes
                ):
                    self.action_space.append(
                        spaces.Tuple(
                            (
                                spaces.Box(
                                    low=-1.0, high=1.0, shape=(2,), dtype=np.float32
                                ),  # eye turns
                                spaces.Discrete(
                                    len(self.move_forward_values)
                                    * len(self.turn_values)
                                ),  # move forward and turn angle
                            )
                        )
                    )
                elif (
                    not self.agent_objects[agent].discrete_actions
                    and not self.agent_objects[agent].use_1dof_eyes
                ):
                    self.action_space.append(
                        spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
                    )
                elif (
                    self.agent_objects[agent].discrete_actions
                    and self.agent_objects[agent].use_1dof_eyes
                ):
                    raise NotImplementedError(
                        "Discrete actions with 1-DOF eyes are not implemented yet."
                    )
                else:
                    self.action_space.append(
                        spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
                    )
            else:
                # Binary eye state
                if (
                    self.agent_objects[agent].discrete_actions
                    and not self.agent_objects[agent].use_1dof_eyes
                ):
                    self.action_space.append(
                        spaces.Tuple(
                            (
                                spaces.Discrete(2),  # Left eye state
                                spaces.Discrete(2),  # Right eye state
                                spaces.Discrete(
                                    len(self.move_forward_values)
                                    * len(self.turn_values)
                                ),  # move forward and turn angle
                            )
                        )
                    )

                elif (
                    not self.agent_objects[agent].discrete_actions
                    and not self.agent_objects[agent].use_1dof_eyes
                ):
                    self.action_space.append(
                        spaces.Tuple(
                            (
                                spaces.Box(
                                    low=-1.0, high=1.0, shape=(2,), dtype=np.float32
                                ),  # move forward and turn angle
                                spaces.Discrete(2 * 2),  # Left * Right eye state
                            )
                        )
                    )
                elif (
                    self.agent_objects[agent].discrete_actions
                    and self.agent_objects[agent].use_1dof_eyes
                ):
                    raise NotImplementedError(
                        "Discrete actions with 1-DOF eyes are not implemented yet."
                    )
                else:
                    self.action_space.append(
                        spaces.Tuple(
                            (
                                spaces.Box(
                                    low=-1.0, high=1.0, shape=(2,), dtype=np.float32
                                ),  # move forward and turn angle
                                spaces.Discrete(
                                    2
                                ),  # Left eye state (right eye mirrors it)
                            )
                        )
                    )

            if self.observation_callback is not None:
                obs_dim = len(self.observation_callback(agent, self))
            else:
                obs_dim = (
                    4 # left/right eye angles (2),  eye fatigue/agent fatigue (2), 
                    + 2 * 2 * self.agent_objects[0].num_rays # num rays * 2 (distance + type) * 2 (for each eye)
                    + self.agent_objects[0].num_actions # action feedback
                )  
            self.observation_space.append(
                spaces.Box(low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32)
            )

        self.share_observation_space = [
            spaces.Box(
                low=0.0, high=1.0, shape=(obs_dim * self.num_agents,), dtype=np.float32
            )
            for _ in range(self.num_agents)
        ]

        self.OBJECT_TYPES = OBJECT_TYPES
        self.reward_params = REWARDS
        self.FISH_CONSTANTS = FISH_CONSTANTS

        self.blur_effect = all_args["blur_effect"]

        # Define area-specific kwargs mapping
        self.arena_kwargs_mapping = {
            ar.UniformArena.__name__: {
                "reset_food_density": all_args.get(
                    "uniform_reset_food_density", 0.0045
                ),
                "max_food_density": all_args.get("uniform_max_food_density", 0.005),
                "step_food_density": all_args.get("uniform_step_food_density", 0.00001),
            },
            ar.PatchyArena.__name__: {
                "reset_food_density": all_args.get("patchy_reset_food_density", 0.015),
                "max_food_density": all_args.get("patchy_max_food_density", 0.02),
                "step_food_density": all_args.get("patchy_step_food_density", 0.001),
            },
            ar.FeederArena.__name__: {
                "reset_food_density": all_args.get("feeder_reset_food_density", 0.015),
                "max_food_density": all_args.get("feeder_max_food_density", 0.02),
                "step_food_density": all_args.get("step_food_density", 0.001),
            },
        }

        self.common_arena_kwargs = {
            "step_food_decay": all_args.get("step_food_decay", 0.0),
            "stockpile_density": all_args.get("stockpile_density", 0.5),
            "food_speed": self.food_speed,
            "food_turn_std": self.food_turn_std,
        }

        self.arena_size_max = ENV_PARAMS["arena_size_max"]
        self.arena_size_min = ENV_PARAMS["arena_size_min"]

        self.shape_reward = REWARDS["shape_reward"]
        self.vergence_deviation_penalty = all_args["vergence_deviation"]
        self.large_move_penalty = all_args.get("large_move_penalty", REWARDS["large_move"])
        self.large_turn_penalty = all_args.get("large_turn_penalty", REWARDS["large_turn"])

        self.curriculum_type = all_args.get("curriculum_type", "fixed_step_with_max")
        self.curriculum_early_end_frac = all_args.get("curriculum_early_end_frac", 0.9)  # only used in time_normalized_step curriculum
        self.fix_food_speed = all_args.get("fix_food_speed", False)

        self.reset(seed=seed)

    def seed(self, seed):
        super().seed(seed)
        self.env_seed = seed  # used in render for tracking env seeds
        # update np_random
        if seed is not None:
            self.np_random = np.random.default_rng(seed=seed)
        elif not hasattr(self, "np_random"):
            self.np_random = np.random.default_rng()

    def _apply_time_normalized_curriculum(self, debug=False):
        assert self.total_num_episodes is not None, "total_num_episodes must be specified for time_normalized_step curriculum."
        effective_total = self.total_num_episodes * self.curriculum_early_end_frac
        progress = (self.current_episode - 1) / effective_total
        progress = np.clip(progress, 0.0, 1.0)
        self.eating_distribution_decay = self.eating_distribution_decay_start + progress * (self.eating_distribution_decay_max - self.eating_distribution_decay_start)
        self.food_speed = progress * self.food_speed_max  # min speed is 0
        self.food_turn_std = progress * self.food_turn_std_max  # min turn is 0
        self.common_arena_kwargs["food_speed"] = self.food_speed
        self.common_arena_kwargs["food_turn_std"] = self.food_turn_std

        REWARDS["collision"] = self.r_collide_start + progress * (self.r_collide_end - self.r_collide_start)

        if debug:
            print(f"Curriculum progress: {progress:.4f}, eating_distribution_decay: {self.eating_distribution_decay:.4f}, food_speed: {self.food_speed:.4f}, food_turn_std: {self.food_turn_std:.4f}, collision reward: {REWARDS['collision']:.4f}")

        # optionally add in scaling for self.min_food_density

    def _apply_fixed_step_with_max_curriculum(self, debug=False):
        if self.eating_distribution_decay < self.eating_distribution_decay_max:
            self.eating_distribution_decay += self.eating_distribution_decay_step
            if debug:
                total_steps = (self.eating_distribution_decay_max - self.eating_distribution_decay_start) / self.eating_distribution_decay_step
                horizon = (self.eating_distribution_decay_max - self.eating_distribution_decay)/self.eating_distribution_decay_step
                print(f"Remaining eating_distribution_decay:  {horizon:<.0f} [/{total_steps:.0f} updates]")

        if self.food_speed < self.food_speed_max:
            self.food_speed += self.food_speed_increase
            if debug:
                if self.food_speed_increase > 0:
                    total_steps = (self.food_speed_max) / self.food_speed_increase 
                    horizon = (self.food_speed_max - self.food_speed)/self.food_speed_increase
                else: 
                    total_steps = 1
                    horizon = 0
                print(f"Remaining food_speed:  {horizon:<.0f} [/{total_steps:.0f} updates]")
            self.common_arena_kwargs["food_speed"] = self.food_speed

        if self.food_turn_std < self.food_turn_std_max:
            self.food_turn_std += self.food_turn_std_increase
            self.common_arena_kwargs["food_turn_std"] = self.food_turn_std
            if debug:
                if self.food_turn_std_increase > 0:
                    total_steps = (self.food_turn_std_max) / self.food_turn_std_increase
                    horizon = (self.food_turn_std_max - self.food_turn_std)/self.food_turn_std_increase
                else:
                    total_steps = 1
                    horizon = 0 
                print(f"Remaining food_turn_std:  {horizon:<.0f} [/{total_steps:.0f} updates]")

        if ((self.r_collide_step > 0 and self.r_collide_start < self.r_collide_end) or
        (self.r_collide_step < 0 and self.r_collide_start > self.r_collide_end)):
            # Step update
            self.r_collide_start += self.r_collide_step
            # Clamp to end value
            if self.r_collide_step > 0:
                self.r_collide_start = min(self.r_collide_start, self.r_collide_end)
            else:
                self.r_collide_start = max(self.r_collide_start, self.r_collide_end)
            # Update reward
            REWARDS["collision"] = self.r_collide_start
            if debug:
                total_steps = abs(self.r_collide_end - REWARDS["collision"]) / abs(self.r_collide_step)
                horizon = abs(self.r_collide_end - self.r_collide_start) / abs(self.r_collide_step)
                print(f"Remaining collision reward: {horizon:<.0f} [/{total_steps:.0f} updates]")

        if debug:
            print(f"Current eating_distribution_decay: {self.eating_distribution_decay:.4f}, food_speed: {self.food_speed:.4f}, food_turn_std: {self.food_turn_std:.4f}, collision reward: {REWARDS['collision']:.4f}")
        # optionally add in scaling for self.min_food_density

    def reset(self, seed=None):
        # Reset internal variables
        if seed is not None or not hasattr(self, "np_random"):
            self.seed(seed)

        self.current_episode += 1

        if not self.is_eval:
            if self.curriculum_type == "time_normalized_step":
                self._apply_time_normalized_curriculum()
            elif self.curriculum_type == "fixed_step_with_max":
                self._apply_fixed_step_with_max_curriculum()

            if self.fix_food_speed:
                self.food_speed = self.food_speed_max
                self.food_turn_std = self.food_turn_std_max

        self.step_count = 0
        self.curr_time = 0
        self.homing_success_counter = 0

        self.arena_size = (
            self.np_random.choice(
                np.arange(
                    int(self.arena_size_min[0]), int(self.arena_size_max[0] + 1)
                )
            ),
            self.np_random.choice(
                np.arange(
                    int(self.arena_size_min[1]), int(self.arena_size_max[1] + 1)
                )
            ),
        )
        
        arena_names, probabilities = zip(*self.arena_generators)
        selected_arena_name = self.np_random.choice(arena_names, p=probabilities)

        # At train time, probabilistically scale food density
        if not self.is_eval:
            if self.train_food_scaling_type == "uniform":
                # storing on object for ease of logging
                self.food_scaling_factor = self.np_random.uniform(
                    self.train_food_scaling_min, self.train_food_scaling_max
                )
            elif self.train_food_scaling_type == "log_uniform":
                self.food_scaling_factor = np.exp(
                    self.np_random.uniform(
                        np.log(self.train_food_scaling_min),
                        np.log(self.train_food_scaling_max),
                    )
                )
            episode_arena_kwargs_mapping = self.arena_kwargs_mapping[
                selected_arena_name
            ].copy()
            for key in ["reset_food_density", "max_food_density", "step_food_density"]:
                episode_arena_kwargs_mapping[key] *= self.food_scaling_factor

            episode_arena_kwargs_mapping["reset_food_density"] = max(
                episode_arena_kwargs_mapping["reset_food_density"],
                self.min_food_density
            )
            episode_arena_kwargs_mapping["max_food_density"] = max(
                episode_arena_kwargs_mapping["max_food_density"],
                self.min_food_density + 0.00001
            )
        else:
            episode_arena_kwargs_mapping = self.arena_kwargs_mapping[
                selected_arena_name
            ]

        # print("Current episode reset food density:", episode_arena_kwargs_mapping["reset_food_density"])

        ## Setup Arena
        # Combine common kwargs with arena-specific kwargs
        arena_kwargs = {
            "min_arena_size": self.arena_size,
            "max_arena_size": self.arena_size,
            **self.common_arena_kwargs,
            **episode_arena_kwargs_mapping,
        }

        # Add any additional arena-specific parameters
        if selected_arena_name == ar.PatchyArena.__name__:
            arena_kwargs.update(
                {
                    "patch_d_mean": 9,
                    "patch_d_var": 2,
                    "reset_patch_density": self.all_args["reset_patch_density"],
                    "step_patch_density": self.all_args["step_patch_density"],
                }
            )
            if not self.is_eval:
                arena_kwargs["reset_patch_density"] *= self.food_scaling_factor
                arena_kwargs["step_patch_density"] *= self.food_scaling_factor
                # testing out varying patch radius... TODO revisit to decide!
                # arena_kwargs["patch_d_mean"] *= self.food_scaling_factor
                # arena_kwargs["patch_d_var"] *= self.food_scaling_factor
        elif selected_arena_name == ar.FeederArena.__name__:
            arena_kwargs.update(
                {
                    "patch_d_mean": 12,
                    "patch_d_var": 3,
                    "patches_per_edge": self.np_random.choice([1, 2]),
                }
            )
        self.arena = self.arena_mapping[selected_arena_name](**arena_kwargs)
        # print("Food speed/std:", self.arena.food_speed, self.arena.food_turn_std)
        arena_seed = self.np_random.integers(10e8)
        self.arena.reset(seed=arena_seed)

        # Initialize random walkers
        self.walkerbots = []
        self.walker_np_random = np.random.default_rng(self.np_random.integers(10e8)) # separate rng for walkerbots so that RNG state remains consistent between runs
        for i in range(self.num_walkerbots):
            wb = WalkerBot(
                arena_size=self.arena_size,
                max_step=self.walker_max_step,
                max_turn=self.walker_max_turn,
                seed=self.walker_np_random.integers(10e8),
                pursuit=self.walker_pursuit,
                agent_perception_radius=self.walker_agent_perception_radius,
                ignore_walls=self.walker_ignore_walls,
            )
            # populate circle params if used
            wb.arena_radius = self.arena.arena_radius
            wb.arena_center = self.arena.center
            wb.reset()  # random init
            self.walkerbots.append(wb)

        for i, agent in enumerate(self.agent_objects):
            # if seed is None:
            #    agent_seed = None
            # else:
            #    agent_seed = seed * 100003 + i * 10007
            agent_seed = self.np_random.integers(10e8)
            # agent_seed = self.env_seed + self.current_episode * 100003 + i * 10007
            agent.arena_size = self.arena_size
            agent.eating_distribution_decay = self.eating_distribution_decay
            agent.arena_radius = self.arena.arena_radius  # Use actual radius
            agent.arena_center = self.arena.center  # Use actual center

            agent.reset(seed=agent_seed)

            # In p_init_closeby fraction of the time, place agents close to each other
            if self.all_args.get("p_init_closeby", 0.0) > 0.0:
                if self.np_random.uniform(0, 1) < self.all_args["p_init_closeby"]:
                    self._init_agents_in_corner(
                        self.agent_objects,
                        self.arena_size,
                        corner=None,  # Picks a random corner
                    )

        if self.reset_callback is not None:
            self.reset_callback(self)
        # observations = [self._next_observation(agent) for agent in self.agent_objects]
        observations = [self._next_observation(agent) for agent in self.agent_objects]
        return observations

    def _resolve_overlaps(self, max_iterations=3):
        #! not used
        """
        Detect and resolve overlapping agents by adjusting their positions (pulling them apart).
        Returns a set of agents that are still overlapping after max iterations.
        """
        overlapping_agents = set()

        # Precompute agent positions and radii
        positions = np.array([agent.position for agent in self.agent_objects])
        agent_radius = self.agent_objects[0].agent_radius
        min_distance = 2 * agent_radius  # Minimum distance to avoid overlap

        for _ in range(max_iterations):
            overlap_occurred = False
            overlapping_agents.clear()

            # Compute distance matrix between agents
            dist_matrix = distance_matrix(positions, positions)

            # Find indices of overlapping pairs
            overlap_indices = np.where(dist_matrix < min_distance)
            unique_pairs = {
                (min(i, j), max(i, j)) for i, j in zip(*overlap_indices) if i != j
            }

            if not unique_pairs:
                break

            overlap_occurred = True
            for i, j in unique_pairs:
                agent_i = self.agent_objects[i]
                agent_j = self.agent_objects[j]
                overlapping_agents.add(agent_i)
                overlapping_agents.add(agent_j)

                # Calculate overlap amount and adjust positions
                distance = dist_matrix[i, j]
                overlap = min_distance - distance
                direction = (positions[i] - positions[j]) / (
                    distance + 1e-8
                )  # Avoid divide by zero
                adjustment = (
                    0.525 * direction * overlap
                )  # Adjust by half the overlap distance (and a little more)

                # Update positions
                positions[i] += adjustment
                positions[j] -= adjustment

                # Ensure agents stay within arena boundaries
                positions[i] = np.clip(
                    positions[i],
                    [agent_radius, agent_radius],
                    [
                        self.arena_size[0] - agent_radius,
                        self.arena_size[1] - agent_radius,
                    ],
                )
                positions[j] = np.clip(
                    positions[j],
                    [agent_radius, agent_radius],
                    [
                        self.arena_size[0] - agent_radius,
                        self.arena_size[1] - agent_radius,
                    ],
                )

            # Update agent positions in the main list
            for idx, agent in enumerate(self.agent_objects):
                agent.position = positions[idx]

        return overlapping_agents

    def step(self, actions):
        self.step_count += 1
        observations = []
        rewards = []
        terminations = []
        infos = [{} for _ in range(self.num_agents)]

        self.arena.step()

        old_agent_positions = np.array([agent.position for agent in self.agent_objects])
        for wb in self.walkerbots:
            wb.step(old_agent_positions)

        # extract the previous agent positions
        # update the agent array in the arena

        agentlike_objects = self.agent_objects + self.walkerbots
        agentlike_positions = [obj.position for obj in agentlike_objects]

        for agent, action in zip(self.agent_objects, actions):
            # agentlike_objects = self.agent_objects + self.walkerbots
            # agentlike_positions = [obj.position for obj in agentlike_objects]

            agent.step(action, self.arena, agentlike_objects)
            # self.arena.update_agent_position(agent.agent_id, agent.position)
            agent.update_projected_sensors(
                agentlike_positions, self.arena.food_positions, self.arena.food_pellets
            )

                # DEBUG
                # print(f"Agent {agent.agent_id} left sensors: {agent.left_eye_sensors}, right sensors: {agent.right_eye_sensors}")
        # overlapping_agents = self._resolve_overlaps()

        # if len(overlapping_agents)>0:
        #     print(f"Overlap detected at step {self.step_count}, overlapping agents: {[agent.agent_id for agent in overlapping_agents]}")

        for agent in self.agent_objects:
            if self.observation_callback is not None:
                observation = self.observation_callback(agent, self)

            agent.has_nearby = self._has_agents_nearby(agent)

            observation = self._next_observation(agent)
            observations.append(observation)

            if self.reward_callback is not None:
                reward = self.reward_callback(agent, self)
            reward = self._calculate_reward(agent, self)
            agent.cumulative_reward += reward
            rewards.append([reward])

            if self.done_callback is not None:
                done = self.done_callback(self)
            done = agent.energy <= 0
            done = False  # don't let agents die
            terminations.append(done)

            if self.info_callback is not None:
                info_agent = self.info_callback(agent, self)
                infos[agent] = info_agent

            # Log state variables for analysis
            infos[agent.agent_id].update(
                {
                    "position": agent.position.copy(),
                    "orientation": agent.orientation,
                    "left_eye_angle": agent.left_eye_angle,
                    "right_eye_angle": agent.right_eye_angle,
                    "energy": agent.energy,
                    "has_nearby": agent.has_nearby,
                    "collided": agent.collided,
                    "detected_food_ids": agent.detected_food_ids.copy(),
                    "eaten_food_ids": agent.food_consumed_ids.copy(),
                    "cumulative_reward": agent.cumulative_reward,
                }
            )

            # --- geometric wall distance ---
            center = self.arena.center
            R = self.arena.arena_radius
            bw = (R - agent.body_radius) - np.linalg.norm(agent.position - center)
            infos[agent.agent_id]["dist_to_wall"] = float(bw)

            # Log per-episode data
            if agent.agent_id == 0:
                food_ids = [pellet.global_index for pellet in self.arena.food_pellets]

                infos[agent.agent_id].update(
                    {
                        "food_positions": self.arena.food_positions.copy(),
                        "food_ids": food_ids,
                    }
                )

                if self.walkerbots:
                    infos[agent.agent_id].update(
                        {
                            "walkerbot_positions": np.array(
                                [wb.position.copy() for wb in self.walkerbots]
                            )
                        }
                    )

                if self.step_count == 1:
                    infos[agent.agent_id].update( # Arena information
                        {
                            "arena_type": self.arena.__class__.__name__,
                            "patch_kwargs": self.arena.patch_kwargs,
                            "arena_size": self.arena_size,
                            "food_speed": self.arena.food_speed,
                            "food_turn_std": self.arena.food_turn_std,
                            "reset_food_density": self.arena.reset_food_density,
                            "max_food_density": self.arena.max_food_density,
                        }
                    )
                    infos[agent.agent_id].update( # Vergence limitations -- used for virtual experiments
                        {
                            "max_left_vergence": agent.max_left_vergence,
                            "max_right_vergence": agent.max_right_vergence,
                            "min_left_vergence": agent.min_left_vergence,
                            "min_right_vergence": agent.min_right_vergence,
                        }
                    )
                    infos[agent.agent_id].update( # Num bots
                        {
                            "num_bots": self.num_walkerbots,
                        }
                    )

        # Don't end episode if run out of food
        # if len(self.arena.food_positions) == 0:
        #     terminations = [True for _ in terminations]

        if self.curr_time >= self.all_args["max_episode_length"]:
            terminations = [True for _ in terminations]

        if self.shared_reward:
            reward = np.sum(rewards)
            rewards = [[reward]] * self.num_agents

        if self.post_step_callback is not None:
            self.post_step_callback(self)

        self.curr_time += 1

        return observations, rewards, terminations, infos

    def _init_agents_in_corner(
        self,
        agents,
        arena_size,
        corner=None,
        spacing=(AGENT_PARAMS["body_radius"] * 3),
    ):
        if corner is None:
            corner = self.np_random.choice(
                ["bottom_left", "bottom_right", "top_left", "top_right"]
            )
        assert corner in [
            "bottom_left",
            "bottom_right",
            "top_left",
            "top_right",
        ], f"Invalid corner specified {corner}"
        base_pos = {
            "bottom_left": np.array([5.0, 5.0]),
            "bottom_right": np.array([arena_size[0] - 5.0, 5.0]),
            "top_left": np.array([5.0, arena_size[1] - 5.0]),
            "top_right": np.array([arena_size[0] - 5.0, arena_size[1] - 5.0]),
        }[corner]

        # TODO: maybe add more heading options
        heading = {
            "bottom_left": np.pi / 2,  # Upward facing
            "bottom_right": np.pi / 2,  # Upward facing
            "top_left": -np.pi / 2,  # Downward facing
            "top_right": -np.pi / 2,  # Downward facing
        }[corner]

        for i, agent in enumerate(agents):
            offset = np.array([i % 2, i // 2]) * spacing
            agent.position = base_pos + offset
            agent.orientation = heading
            agent.trajectory = [agent.position.copy()]

    def _next_observation(self, agent, feedback_action=True):
        agent_xy = agent.position

        action_observations = (
            agent.last_action if feedback_action else [0] * agent.num_actions
        )

        sensor_observations = np.concatenate(
            (agent.left_eye_sensors.flatten(), agent.right_eye_sensors.flatten())
        )

        if self.blur_effect:
            if agent.binary_eye_state:
                if (
                    agent.left_eye_state != agent.prev_left_eye_state
                    or agent.right_eye_state != agent.prev_right_eye_state
                ):
                    sensor_observations = np.zeros_like(sensor_observations)

        # print(sensor_observations, flush=True)
        observation = np.concatenate(
            [
                sensor_observations,
                np.array([agent.left_eye_angle, agent.right_eye_angle]),
                np.array(action_observations).flatten(),
                [agent.fatigue_count if agent.use_fatigue else 0],
                [agent.eye_fatigue if agent.use_eye_fatigue else 0],
            ]
        )
        return observation

    def _line_segment_intersection(self, p1, p2, q1, q2):
        # Check if two line segments (p1, p2) and (q1, q2) intersect
        # Returns the distance from p1 to the intersection point, or None if no intersection
        r = p2 - p1
        s = np.array(q2) - np.array(q1)
        det = r[0] * s[1] - r[1] * s[0]
        if det == 0:
            return None
        t = -((q1[1] - p1[1]) * s[0] - (q1[0] - p1[0]) * s[1]) / det
        u = -((q1[1] - p1[1]) * r[0] - (q1[0] - p1[0]) * r[1]) / det
        if 0 <= t <= 1 and 0 <= u <= 1:
            return np.linalg.norm(p1 + t * r - p1)
        return None

    def _line_segment_circle_intersection(self, p1, p2, center, radius):
        # Check if a line segment (p1, p2) intersects with a circle (center, radius)
        # Returns the distance from p1 to the intersection point, or None if no intersection
        v = p2 - p1
        a = v.dot(v)
        b = 2 * v.dot(p1 - center)
        c = p1.dot(p1) + center.dot(center) - 2 * p1.dot(center) - radius**2
        disc = b**2 - 4 * a * c
        if disc < 0:
            return None
        sqrt_disc = np.sqrt(disc)
        t1 = (-b + sqrt_disc) / (2 * a)
        t2 = (-b - sqrt_disc) / (2 * a)
        if 0 <= t1 <= 1:
            return t1 * np.linalg.norm(v)
        if 0 <= t2 <= 1:
            return t2 * np.linalg.norm(v)
        return None

    def _has_agents_nearby(self, agent):
        if len(self.agent_objects) <= 1:
            return False
        agent_distances = cdist(
            [agent.position],
            [
                other_agent.position
                for other_agent in self.agent_objects
                if other_agent != agent
            ],
        )[0]
        return np.any(agent_distances <= agent.fish_sensing_radius)

    def _calculate_reward(self, agent, arena):
        # Food/eating related
        reward = 0
        if self.arena.food_positions.size == 0:  # No food remaining
            pass
        else:
            food_distances = cdist([agent.position], self.arena.food_positions)[0]
            nearest_food_distance = np.min(food_distances)

            if REWARDS["max_align_reward"] > 0:
                # Calculate alignment reward if enabled
                if nearest_food_distance < agent.food_sensing_radius:
                    nearest_food_pos = self.arena.food_positions[
                        np.argmin(food_distances)
                    ]
                    to_food_vector = nearest_food_pos - agent.position
                    to_food_angle = np.arctan2(to_food_vector[1], to_food_vector[0])
                    angle_diff = normalize_angle(to_food_angle - agent.orientation)
                    self.to_food_angle = angle_diff  # Store for debugging
                    reward += (
                        np.cos(angle_diff)
                        * cfg.REWARDS["max_align_reward"]
                        # * (1 - nearest_food_distance / agent.food_sensing_radius)
                    )

            # if agent.previous_food_distance is not None:
            # Calculate reward based on both distance and orientation to food
            if self.shape_reward and agent.previous_food_distance is not None:
                # Distance-based reward
                distance_reward = (
                    agent.previous_food_distance - nearest_food_distance
                ) / np.sum(arena.arena_size)

                # Test
                alpha = 0.0  # arbitrary
                weight = np.exp(-alpha * nearest_food_distance)
                distance_reward = 0.01 * (
                    agent.previous_food_distance - nearest_food_distance
                )
                distance_reward *= weight

                reward += distance_reward

                # # Find the nearest food position
                # nearest_food_idx = np.argmin(food_distances)
                # nearest_food_pos = self.arena.food_positions[nearest_food_idx]

                # # Calculate angle from agent to nearest food
                # food_direction = nearest_food_pos - agent.position
                # food_angle = np.arctan2(food_direction[1], food_direction[0])

                # # Calculate angle difference between agent orientation and food direction
                # angle_diff = food_angle - agent.orientation
                # # Normalize angle difference to [-π, π]
                # angle_diff = np.arctan2(np.sin(angle_diff), np.cos(angle_diff))

                # # Orientation reward: higher when agent is facing toward food
                # # cos(angle_diff) gives 1 when perfectly aligned, -1 when opposite
                # orientation_reward = np.cos(angle_diff) * 0.01  # Scale the orientation reward

                # reward = distance_reward + orientation_reward

            agent.previous_food_distance = nearest_food_distance
            reward += REWARDS["eat"] * len(agent.curr_food_consumed)

        # Collision penalty
        if agent.collided:
            reward += REWARDS[
                "collision"
            ]  # Subtract from the reward if a collision occurred

        # Energy related
        # reward += -10 if agent.energy < self.energy_minimum else 0
        if self.penalize_move_threshold is not None:
            if agent.move_forward > self.penalize_move_threshold:
                reward += self.large_move_penalty * (
                    agent.move_forward - self.penalize_move_threshold
                )

        if self.penalize_turn_threshold is not None:
            excess_turn = abs(agent.turn_angle) - self.penalize_turn_threshold
            if excess_turn > 0:
                reward += self.large_turn_penalty * excess_turn
                
        if agent.use_fatigue:
            # Fatigue penalty
            reward += agent.fatigue_count * REWARDS["fatigue"]

        if agent.use_eye_fatigue and agent.eye_fatigue_penalty > 0:
            eye_fatigue_threshold = 2 * (
                FISH_CONSTANTS["min_right_vergence"]
                - FISH_CONSTANTS["max_right_vergence"]
            )
            reward += (
                np.max(agent.eye_fatigue - eye_fatigue_threshold, 0)
                * agent.eye_fatigue_penalty
            )

        reward += (
            np.abs(agent.left_eye_angle - FISH_CONSTANTS["min_left_vergence"])
            * self.vergence_deviation_penalty
        )
        reward += (
            np.abs(agent.right_eye_angle - FISH_CONSTANTS["min_right_vergence"])
            * self.vergence_deviation_penalty
        )

        if agent.binary_eye_state:
            if agent.left_eye_state != agent.prev_left_eye_state:
                reward += REWARDS["switching_vergence"]
            if agent.right_eye_state != agent.prev_right_eye_state:
                reward += REWARDS["switching_vergence"]

        return reward

    def _plot_projected_sensing(self, ax):
        for agent in self.agent_objects:
            # print(f"Agent {agent.agent_id} left sensors: {agent.left_eye_sensors}, right sensors: {agent.right_eye_sensors}")
            left_eye_pos, right_eye_pos = (
                agent._get_eye_positions()
            )

            # Plot projected sensor rays for both eyes
            for eye_name, sensors, eye_pos in [
                ("left", agent.left_eye_sensors, left_eye_pos),
                ("right", agent.right_eye_sensors, right_eye_pos),
            ]:
                eye_angle = (
                    agent.left_eye_angle
                    if eye_name == "left"
                    else agent.right_eye_angle
                )

                for i, (distance, object_type) in enumerate(sensors):
                    if distance > 0:  # Only plot if something is detected
                        # Calculate the angle for this sensor
                        sector_center_angle = (
                            eye_angle
                            - agent.perception_field / 2
                            + (i + 0.5) * agent.projected_sector_angle
                        )
                        global_angle = sector_center_angle + agent.orientation

                        # Calculate the end point based on detected distance and object type
                        if object_type == OBJECT_TYPES["WALL"]:
                            continue #TODO: skip wall for now
                            max_range = (
                                agent.ray_length
                            )  # NOTE: always use same ray_length for normalization
                            color = "orange"
                        elif object_type == OBJECT_TYPES["FOOD"]:
                            max_range = agent.ray_length
                            color = "green"
                        elif object_type == OBJECT_TYPES["AGENT"]:
                            max_range = agent.ray_length
                            color = "blue"
                        else:
                            continue  # Skip if no object detected

                        # Convert normalized distance back to actual distance
                        actual_distance = (1 - distance) * max_range
                        end_point = eye_pos + actual_distance * np.array(
                            [np.cos(global_angle), np.sin(global_angle)]
                        )

                        # Plot the sensing ray
                        # ax.plot(
                        #     [eye_pos[0], end_point[0]],
                        #     [eye_pos[1], end_point[1]],
                        #     linestyle="dotted",
                        #     color=color,
                        #     linewidth=1,
                        #     alpha=0.8,
                        # )

                    elif (
                        distance == UNKNOWN_DIST
                    ):  # detected by one eye but no distance info
                        # Plot a dashed line to indicate detection without distance
                        sector_center_angle = (
                            eye_angle
                            - agent.perception_field / 2
                            + (i + 0.5) * agent.projected_sector_angle
                        )
                        global_angle = sector_center_angle + agent.orientation

                        end_point = eye_pos + agent.ray_length * np.array(
                            [np.cos(global_angle), np.sin(global_angle)]
                        )
                        # if object_type == OBJECT_TYPES["FOOD"]:
                        #     ax.plot(
                        #         [eye_pos[0], end_point[0]],
                        #         [eye_pos[1], end_point[1]],
                        #         linestyle="dashed",
                        #         color="green",
                        #         linewidth=1,
                        #         alpha=0.8,
                        #     )
                        # else:
                        #     ax.plot(
                        #         [eye_pos[0], end_point[0]],
                        #         [eye_pos[1], end_point[1]],
                        #         linestyle="dashed",
                        #         color="blue",
                        #         linewidth=1,
                        #         alpha=0.8,
                        #     )

    def _plot_speed(self, ax):
        # Plot speed over time for each agent
        if not hasattr(self, "speed_history"):
            self.speed_history = {i: [] for i in range(self.num_agents)}
            self.eating_events = {i: [] for i in range(self.num_agents)}
            self.food_in_range = {i: [] for i in range(self.num_agents)}
            self.miss_events = {i: [] for i in range(self.num_agents)}
            self.previous_detected_food_ids = {i: set() for i in range(self.num_agents)}

        # Update histories
        for i, agent in enumerate(self.agent_objects):
            # Calculate current speed (move_forward is the speed parameter)
            current_speed = agent.move_forward * FISH_CONSTANTS["max_speed"]
            self.speed_history[i].append(current_speed)

            # Check if agent is eating (has consumed food this step)
            is_eating = len(agent.curr_food_consumed) > 0
            self.eating_events[i].append(is_eating)

            # Check if food is detected by agent's sensors
            food_detected = len(agent.detected_food_ids) > 0
            self.food_in_range[i].append(food_detected)

            # Check for miss events (food ID left detection but wasn't eaten)
            current_detected = set(agent.detected_food_ids)
            previous_detected = self.previous_detected_food_ids[i]
            eaten_ids = set(agent.food_consumed_ids)

            # Find IDs that were detected last step but not this step
            lost_ids = previous_detected - current_detected
            # Check if any lost IDs were not eaten
            missed_ids = lost_ids - eaten_ids
            is_missing = len(missed_ids) > 0
            self.miss_events[i].append(is_missing)

            # Update previous detected food IDs
            self.previous_detected_food_ids[i] = current_detected

        ax.clear()
        ax.set_title("Agent Speed (Last 50 Steps)")
        ax.set_ylabel("Speed (mm/s)")

        has_data = False
        # Plot speed history for each agent (last 50 steps)
        for i in range(self.num_agents):
            if len(self.speed_history[i]) > 1:
                has_data = True
                # Get last 50 data points
                recent_speed = self.speed_history[i][-50:]
                recent_eating = self.eating_events[i][-50:]
                recent_food_range = self.food_in_range[i][-50:]
                recent_miss = self.miss_events[i][-50:]

                time_steps = range(len(recent_speed))
                ax.plot(time_steps, recent_speed, label=f"Agent {i}", linewidth=2)

                # Mark eating events
                eating_steps = [t for t, eating in enumerate(recent_eating) if eating]
                if eating_steps:
                    eating_speeds = [recent_speed[t] for t in eating_steps]
                    ax.scatter(
                        eating_steps,
                        eating_speeds,
                        color="green",
                        s=50,
                        marker="o",
                        label=f"Eating (Agent {i})" if i == 0 else "",
                        zorder=5,
                    )

                # Mark miss events
                miss_steps = [t for t, missing in enumerate(recent_miss) if missing]
                if miss_steps:
                    miss_speeds = [recent_speed[t] for t in miss_steps]
                    ax.scatter(
                        miss_steps,
                        miss_speeds,
                        color="red",
                        s=50,
                        marker="x",
                        label=f"Miss (Agent {i})" if i == 0 else "",
                        zorder=5,
                    )

                # Add shaded intervals for food detection
                food_intervals = []
                current_interval_start = None
                for t, in_range in enumerate(recent_food_range):
                    if in_range and current_interval_start is None:
                        current_interval_start = t
                    elif not in_range and current_interval_start is not None:
                        food_intervals.append((current_interval_start, t))
                        current_interval_start = None
                # Handle case where food detection continues to the end
                if current_interval_start is not None:
                    food_intervals.append(
                        (current_interval_start, len(recent_food_range) - 1)
                    )

                # Shade the intervals
                # for start, end in food_intervals:
                #     ax.axvspan(
                #         start,
                #         end,
                #         alpha=0.2,
                #         color="gray",
                #         label=(
                #             f"Food detected (Agent {i})"
                #             if i == 0 and start == food_intervals[0][0]
                #             else ""
                #         ),
                #     )

        if has_data:
            ax.legend(loc="upper left", fontsize="small")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(
            0,
            max(self.agent_objects[0].move_forward_values)
            * 1.1
            * FISH_CONSTANTS["max_speed"],
        )

    def _plot_turn_angle(self, ax):
        # Plot turn angle over time for each agent
        if not hasattr(self, "turn_angle_history"):
            self.turn_angle_history = {i: [] for i in range(self.num_agents)}

        # Update turn angle history
        for i, agent in enumerate(self.agent_objects):
            # Store the turn angle directly
            self.turn_angle_history[i].append(agent.turn_angle)

        ax.clear()
        ax.set_title("Agent Turn Angle (Last 50 Steps)")
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Turn Angle")

        has_data = False
        # Plot turn angle history for each agent (last 50 steps)
        for i in range(self.num_agents):
            if len(self.turn_angle_history[i]) > 1:
                has_data = True
                # Get last 50 data points
                recent_turn_angle = self.turn_angle_history[i][-50:]
                recent_eating = self.eating_events[i][-50:]
                recent_food_range = self.food_in_range[i][-50:]
                recent_miss = self.miss_events[i][-50:]

                time_steps = range(len(recent_turn_angle))
                ax.plot(time_steps, recent_turn_angle, label=f"Agent {i}", linewidth=2)

                # Mark eating events
                eating_steps = [t for t, eating in enumerate(recent_eating) if eating]
                if eating_steps:
                    eating_turn_angles = [recent_turn_angle[t] for t in eating_steps]
                    ax.scatter(
                        eating_steps,
                        eating_turn_angles,
                        color="green",
                        s=50,
                        marker="o",
                        label=f"Eating (Agent {i})" if i == 0 else "",
                        zorder=5,
                    )

                # Mark miss events
                miss_steps = [t for t, missing in enumerate(recent_miss) if missing]
                if miss_steps:
                    miss_turn_angles = [recent_turn_angle[t] for t in miss_steps]
                    ax.scatter(
                        miss_steps,
                        miss_turn_angles,
                        color="red",
                        s=50,
                        marker="x",
                        label=f"Miss (Agent {i})" if i == 0 else "",
                        zorder=5,
                    )

                # Add shaded intervals for food detection
                food_intervals = []
                current_interval_start = None
                for t, in_range in enumerate(recent_food_range):
                    if in_range and current_interval_start is None:
                        current_interval_start = t
                    elif not in_range and current_interval_start is not None:
                        food_intervals.append((current_interval_start, t))
                        current_interval_start = None
                # Handle case where food detection continues to the end
                if current_interval_start is not None:
                    food_intervals.append(
                        (current_interval_start, len(recent_food_range) - 1)
                    )

                # Shade the intervals
                # for start, end in food_intervals:
                #     ax.axvspan(
                #         start,
                #         end,
                #         alpha=0.2,
                #         color="gray",
                #         label=(
                #             f"Food detected (Agent {i})"
                #             if i == 0 and start == food_intervals[0][0]
                #             else ""
                #         ),
                #     )

        # if has_data:
        #     ax.legend(loc='upper left', fontsize='small')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-1.1, 1.1)  # Turn angle is normalized between -1 and 1

    def _plot_vergence(self, ax):
        # Plot vergence angle over time for each agent
        if not hasattr(self, "vergence_history"):
            self.vergence_history = {i: [] for i in range(self.num_agents)}
            self.left_eye_history = {i: [] for i in range(self.num_agents)}
            self.right_eye_history = {i: [] for i in range(self.num_agents)}

        # Update vergence history
        for i, agent in enumerate(self.agent_objects):
            # Calculate vergence angle
            vergence_angle = (
                (agent.left_eye_angle - agent.right_eye_angle + agent.perception_field)
                * 180
                / np.pi
            )
            self.vergence_history[i].append(vergence_angle)

            # Store individual eye angles in degrees
            self.left_eye_history[i].append(-1 * agent.left_eye_angle * 180 / np.pi)
            self.right_eye_history[i].append(agent.right_eye_angle * 180 / np.pi)

        ax.clear()
        ax.set_title("Agent Vergence Angles (Last 50 Steps)")
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Vergence Angle (degrees)")

        # Plot horizontal dotted lines for max and min vergence angles
        max_vergence = (
            (
                self.agent_objects[0].max_left_vergence
                - self.agent_objects[0].max_right_vergence
                + self.agent_objects[0].perception_field
            )
            * 180
            / np.pi
        )
        min_vergence = (
            (
                self.agent_objects[0].min_left_vergence
                - self.agent_objects[0].min_right_vergence
                + self.agent_objects[0].perception_field
            )
            * 180
            / np.pi
        )

        ax.axhline(
            y=max_vergence, color="red", linestyle="--", alpha=0.7, label="Max Vergence"
        )
        ax.axhline(
            y=min_vergence, color="red", linestyle="--", alpha=0.7, label="Min Vergence"
        )

        # Plot horizontal dotted lines for individual eye angle limits
        # max_left_eye = self.agent_objects[0].max_left_vergence * 180 / np.pi
        # min_left_eye = self.agent_objects[0].min_left_vergence * 180 / np.pi
        # max_right_eye = self.agent_objects[0].max_right_vergence * 180 / np.pi
        # min_right_eye = self.agent_objects[0].min_right_vergence * 180 / np.pi

        # ax.axhline(y=max_left_eye, color='blue', linestyle=':', alpha=0.5, label='Max Left Eye')
        # ax.axhline(y=min_left_eye, color='blue', linestyle=':', alpha=0.5, label='Min Left Eye')
        # ax.axhline(y=max_right_eye, color='orange', linestyle=':', alpha=0.5, label='Max Right Eye')
        # ax.axhline(y=min_right_eye, color='orange', linestyle=':', alpha=0.5, label='Min Right Eye')

        has_data = False
        # Plot vergence history for each agent (last 50 steps)
        for i in range(self.num_agents):
            if len(self.vergence_history[i]) > 1:
                has_data = True
                # Get last 50 data points
                recent_vergence = self.vergence_history[i][-50:]
                recent_left_eye = self.left_eye_history[i][-50:]
                recent_right_eye = self.right_eye_history[i][-50:]
                recent_eating = self.eating_events[i][-50:]
                recent_food_range = self.food_in_range[i][-50:]
                recent_miss = self.miss_events[i][-50:]

                time_steps = range(len(recent_vergence))
                ax.plot(
                    time_steps,
                    recent_vergence,
                    label=f"Vergence Agent {i}",
                    linewidth=2,
                )
                ax.plot(
                    time_steps,
                    recent_left_eye,
                    label=f"Left Eye Agent {i}",
                    linestyle=":",
                    alpha=0.7,
                    color="blue",
                )
                ax.plot(
                    time_steps,
                    recent_right_eye,
                    label=f"Right Eye Agent {i}",
                    linestyle=":",
                    alpha=0.7,
                    color="orange",
                )

                # Mark eating events
                eating_steps = [t for t, eating in enumerate(recent_eating) if eating]
                if eating_steps:
                    eating_vergence = [recent_vergence[t] for t in eating_steps]
                    ax.scatter(
                        eating_steps,
                        eating_vergence,
                        color="green",
                        s=50,
                        marker="o",
                        label=f"Eating (Agent {i})" if i == 0 else "",
                        zorder=5,
                    )

                # Mark miss events
                miss_steps = [t for t, missing in enumerate(recent_miss) if missing]
                if miss_steps:
                    miss_vergence = [recent_vergence[t] for t in miss_steps]
                    ax.scatter(
                        miss_steps,
                        miss_vergence,
                        color="red",
                        s=50,
                        marker="x",
                        label=f"Miss (Agent {i})" if i == 0 else "",
                        zorder=5,
                    )

                # Add shaded intervals for food detection
                food_intervals = []
                current_interval_start = None
                for t, in_range in enumerate(recent_food_range):
                    if in_range and current_interval_start is None:
                        current_interval_start = t
                    elif not in_range and current_interval_start is not None:
                        food_intervals.append((current_interval_start, t))
                        current_interval_start = None
                # Handle case where food detection continues to the end
                if current_interval_start is not None:
                    food_intervals.append(
                        (current_interval_start, len(recent_food_range) - 1)
                    )

                # Shade the intervals
                # for start, end in food_intervals:
                #     ax.axvspan(
                #         start,
                #         end,
                #         alpha=0.2,
                #         color="gray",
                #         label=(
                #             f"Food detected (Agent {i})"
                #             if i == 0 and start == food_intervals[0][0]
                #             else ""
                #         ),
                #     )

        # if has_data:
        #     ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_fatigue(self, ax):
        # Plot fatigue over time for each agent
        if not hasattr(self, "fatigue_history"):
            self.fatigue_history = {i: [] for i in range(self.num_agents)}

        # Update fatigue history
        for i, agent in enumerate(self.agent_objects):
            self.fatigue_history[i].append(agent.fatigue_count)

        ax.clear()
        ax.set_title("Agent Fatigue (Last 50 Steps)")
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Fatigue Count")

        has_data = False
        # Plot fatigue history for each agent (last 50 steps)
        for i in range(self.num_agents):
            if len(self.fatigue_history[i]) > 1:
                has_data = True
                # Get last 50 data points
                recent_fatigue = self.fatigue_history[i][-50:]
                recent_eating = self.eating_events[i][-50:]
                recent_food_range = self.food_in_range[i][-50:]
                recent_miss = self.miss_events[i][-50:]

                time_steps = range(len(recent_fatigue))
                ax.plot(time_steps, recent_fatigue, label=f"Agent {i}", linewidth=2)

                # Mark eating events
                eating_steps = [t for t, eating in enumerate(recent_eating) if eating]
                if eating_steps:
                    eating_fatigue = [recent_fatigue[t] for t in eating_steps]
                    ax.scatter(
                        eating_steps,
                        eating_fatigue,
                        color="green",
                        s=50,
                        marker="o",
                        label=f"Eating (Agent {i})" if i == 0 else "",
                        zorder=5,
                    )

                # Mark miss events
                miss_steps = [t for t, missing in enumerate(recent_miss) if missing]
                if miss_steps:
                    miss_fatigue = [recent_fatigue[t] for t in miss_steps]
                    ax.scatter(
                        miss_steps,
                        miss_fatigue,
                        color="red",
                        s=50,
                        marker="x",
                        label=f"Miss (Agent {i})" if i == 0 else "",
                        zorder=5,
                    )

                # Add shaded intervals for food detection
                food_intervals = []
                current_interval_start = None
                for t, in_range in enumerate(recent_food_range):
                    if in_range and current_interval_start is None:
                        current_interval_start = t
                    elif not in_range and current_interval_start is not None:
                        food_intervals.append((current_interval_start, t))
                        current_interval_start = None
                # Handle case where food detection continues to the end
                if current_interval_start is not None:
                    food_intervals.append(
                        (current_interval_start, len(recent_food_range) - 1)
                    )

                # Shade the intervals
                # for start, end in food_intervals:
                #     ax.axvspan(
                #         start,
                #         end,
                #         alpha=0.2,
                #         color="gray",
                #         label=(
                #             f"Food detected (Agent {i})"
                #             if i == 0 and start == food_intervals[0][0]
                #             else ""
                #         ),
                #     )

        ax.grid(True, alpha=0.3)

    def _plot_food_angle(self, ax):
        # Plot food angle over time for each agent
        if not hasattr(self, "food_angle_history"):
            self.food_angle_history = {i: [] for i in range(self.num_agents)}

        # Update food angle history
        for i, agent in enumerate(self.agent_objects):
            # Get the food angle from the environment (stored during reward calculation)
            food_angle = (
                getattr(self, "to_food_angle", 0) if i == 0 else 0
            )  # Only track for first agent
            self.food_angle_history[i].append(food_angle)

        ax.clear()
        ax.set_title("Agent Food Angle (Last 50 Steps)")
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Angle to Food (radians)")

        has_data = False
        # Plot food angle history for each agent (last 50 steps)
        for i in range(self.num_agents):
            if len(self.food_angle_history[i]) > 1:
                has_data = True
                # Get last 50 data points
                recent_food_angle = self.food_angle_history[i][-50:]
                recent_eating = self.eating_events[i][-50:]
                recent_food_range = self.food_in_range[i][-50:]
                recent_miss = self.miss_events[i][-50:]

                time_steps = range(len(recent_food_angle))
                ax.plot(time_steps, recent_food_angle, label=f"Agent {i}", linewidth=2)

                # Mark eating events
                eating_steps = [t for t, eating in enumerate(recent_eating) if eating]
                if eating_steps:
                    eating_angles = [recent_food_angle[t] for t in eating_steps]
                    ax.scatter(
                        eating_steps,
                        eating_angles,
                        color="green",
                        s=50,
                        marker="o",
                        label=f"Eating (Agent {i})" if i == 0 else "",
                        zorder=5,
                    )

                # Mark miss events
                miss_steps = [t for t, missing in enumerate(recent_miss) if missing]
                if miss_steps:
                    miss_angles = [recent_food_angle[t] for t in miss_steps]
                    ax.scatter(
                        miss_steps,
                        miss_angles,
                        color="red",
                        s=50,
                        marker="x",
                        label=f"Miss (Agent {i})" if i == 0 else "",
                        zorder=5,
                    )

                # Add shaded intervals for food detection
                food_intervals = []
                current_interval_start = None
                for t, in_range in enumerate(recent_food_range):
                    if in_range and current_interval_start is None:
                        current_interval_start = t
                    elif not in_range and current_interval_start is not None:
                        food_intervals.append((current_interval_start, t))
                        current_interval_start = None
                # Handle case where food detection continues to the end
                if current_interval_start is not None:
                    food_intervals.append(
                        (current_interval_start, len(recent_food_range) - 1)
                    )

                # Shade the intervals
                # for start, end in food_intervals:
                #     ax.axvspan(
                #         start,
                #         end,
                #         alpha=0.2,
                #         color="gray",
                #         label=(
                #             f"Food detected (Agent {i})"
                #             if i == 0 and start == food_intervals[0][0]
                #             else ""
                #         ),
                #     )

        # Add horizontal lines for reference angles
        ax.axhline(
            y=0, color="black", linestyle="-", alpha=0.3, label="Perfect alignment"
        )
        ax.axhline(y=np.pi / 2, color="red", linestyle="--", alpha=0.5, label="90° off")
        ax.axhline(y=-np.pi / 2, color="red", linestyle="--", alpha=0.5)
        ax.axhline(y=np.pi, color="red", linestyle=":", alpha=0.5, label="180° off")
        ax.axhline(y=-np.pi, color="red", linestyle=":", alpha=0.5)

        ax.grid(True, alpha=0.3)
        ax.set_ylim(-np.pi, np.pi)

    def _plot_eye_fatigue(self, ax):
        # Plot fatigue over time for each agent
        if not hasattr(self, "eye_fatigue_history"):
            self.eye_fatigue_history = {i: [] for i in range(self.num_agents)}

        # Update fatigue history
        for i, agent in enumerate(self.agent_objects):
            self.eye_fatigue_history[i].append(agent.eye_fatigue)

        ax.clear()
        ax.set_title("Agent Eye Fatigue (Last 50 Steps)")
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Eye Fatigue")

        has_data = False
        # Plot fatigue history for each agent (last 50 steps)
        for i in range(self.num_agents):
            if len(self.eye_fatigue_history[i]) > 1:
                has_data = True
                # Get last 50 data points
                recent_fatigue = self.eye_fatigue_history[i][-50:]
                recent_eating = self.eating_events[i][-50:]
                recent_food_range = self.food_in_range[i][-50:]
                recent_miss = self.miss_events[i][-50:]

                time_steps = range(len(recent_fatigue))
                ax.plot(time_steps, recent_fatigue, label=f"Agent {i}", linewidth=2)

                # Mark eating events
                eating_steps = [t for t, eating in enumerate(recent_eating) if eating]
                if eating_steps:
                    eating_fatigue = [recent_fatigue[t] for t in eating_steps]
                    ax.scatter(
                        eating_steps,
                        eating_fatigue,
                        color="green",
                        s=50,
                        marker="o",
                        label=f"Eating (Agent {i})" if i == 0 else "",
                        zorder=5,
                    )

                # Mark miss events
                miss_steps = [t for t, missing in enumerate(recent_miss) if missing]
                if miss_steps:
                    miss_fatigue = [recent_fatigue[t] for t in miss_steps]
                    ax.scatter(
                        miss_steps,
                        miss_fatigue,
                        color="red",
                        s=50,
                        marker="x",
                        label=f"Miss (Agent {i})" if i == 0 else "",
                        zorder=5,
                    )

                # Add shaded intervals for food detection
                food_intervals = []
                current_interval_start = None
                for t, in_range in enumerate(recent_food_range):
                    if in_range and current_interval_start is None:
                        current_interval_start = t
                    elif not in_range and current_interval_start is not None:
                        food_intervals.append((current_interval_start, t))
                        current_interval_start = None
                # Handle case where food detection continues to the end
                if current_interval_start is not None:
                    food_intervals.append(
                        (current_interval_start, len(recent_food_range) - 1)
                    )

                # Shade the intervals
                # for start, end in food_intervals:
                #     ax.axvspan(
                #         start,
                #         end,
                #         alpha=0.2,
                #         color="gray",
                #         label=(
                #             f"Food detected (Agent {i})"
                #             if i == 0 and start == food_intervals[0][0]
                #             else ""
                #         ),
                #     )

        # Draw horizontal line at the threshold
        eye_fatigue_threshold = 2 * (
            FISH_CONSTANTS["min_right_vergence"]
            - FISH_CONSTANTS["max_right_vergence"]
        )
        ax.axhline(
            y=eye_fatigue_threshold,
            color="red",
            linestyle="--",
            alpha=0.7,
            label="Eye Fatigue Threshold",
        )
        ax.grid(True, alpha=0.3)

    def render(self, mode=None):
        buffer_area = 0

        if REWARDS["max_align_reward"] > 0:
            aux_plot_funcs = {
                "speed": self._plot_speed,
                "vergence": self._plot_vergence,
                # "fatigue": self._plot_fatigue,
                "turn_angle": self._plot_turn_angle,
                # "eye_fatigue": self._plot_eye_fatigue,
                "food angle": self._plot_food_angle,
                "local_view": self._plot_local_view,
            }
        else:
            aux_plot_funcs = {
                "speed": self._plot_speed,
                "vergence": self._plot_vergence,
                # "fatigue": self._plot_fatigue,
                "turn_angle": self._plot_turn_angle,
                # "eye_fatigue": self._plot_eye_fatigue,
                "local_view": self._plot_local_view,
            }
        num_aux_subplots = aux_plot_funcs.keys().__len__()
        total_subplots = 1 + num_aux_subplots  # main plot + auxs

        # Create the figure and axes if not already created or if auxs changed
        if self.fig is None or len(self.axes) != total_subplots:
            self.fig = plt.figure(figsize=ENV_PARAMS["render_figsize"], constrained_layout=True)

            # Create grid with main plot on left and aux plots on right
            gs = self.fig.add_gridspec(
                max(1, num_aux_subplots),
                2,
                width_ratios=[1.9, 1],
                height_ratios=[1] * max(1, num_aux_subplots),
                hspace=-0.1,   # vertical spacing between right-column subplots
                wspace=0.0,   # space between main (left) and aux (right)
            )

            # Main plot spans all rows in the left column
            self.ax1 = self.fig.add_subplot(gs[:, 0])

            # Aux plots are stacked vertically in the right column
            self.aux_axes = []
            for i in range(num_aux_subplots):
                ax = self.fig.add_subplot(gs[i, 1])
                self.aux_axes.append(ax)

            # Store axes for length comparison
            self.axes = [self.ax1] + self.aux_axes

        #     # Calculate center offset to center arena in fixed viewport
        #     max_size = max(self.arena_size_max)
        #     center_x = max_size / 2
        #     center_y = max_size / 2

        #     arena_center_x = self.arena_size[0] / 2
        #     arena_center_y = self.arena_size[1] / 2

        #     offset_x = center_x - arena_center_x
        #     offset_y = center_y - arena_center_y

        #     self.ax1.set_xlim(
        #         -offset_x - buffer_area, max_size - offset_x + buffer_area
        #     )
        #     self.ax1.set_ylim(
        #         -offset_y - buffer_area, max_size - offset_y + buffer_area
        #     )
        # self.ax1.set_aspect("equal", "box")

        self.ax1.clear()
        energy_values = [f"{agent.energy:.1f}" for agent in self.agent_objects]
        cumulative_rewards = [
            f"{agent.cumulative_reward:.3f}" for agent in self.agent_objects
        ]
        playback_speed = ENV_PARAMS["fps_video"] / ENV_PARAMS["fps_sim"]
        self.ax1.set_title(
            f"Step: {self.step_count}\nEnergy: {energy_values}\nCumu Reward: {cumulative_rewards}\nPlayback Speed: {playback_speed:.2f}",
            fontsize=9,
        )

        # Draw the arena boundaries
        # Draw circular arena boundary
        arena_circle = plt.Circle(
            self.arena.center,
            self.arena.arena_radius,
            edgecolor="black",
            fill=False,
            linewidth=2,
            clip_on=False,
        )
        self.ax1.add_patch(arena_circle)

        V = max(self.arena_size_max)  # e.g., width/height of your largest arena
        pad = 0.0                     # optional constant padding in data units
        half = V / 2 + pad
        cx, cy = self.arena.center

        self.ax1.set_aspect("equal", adjustable="box")
        self.ax1.set_anchor("C")  # keep the data rectangle centered in its axes box
        self.ax1.set_autoscale_on(False)

        # Center the fixed-size viewport on the current arena center
        self.ax1.set_xlim(cx - half, cx + half)
        self.ax1.set_ylim(cy - half, cy + half)

        # Remove the rectangular axes frame/spines for circular arenas
        self.ax1.spines["top"].set_visible(False)
        self.ax1.spines["right"].set_visible(False)
        self.ax1.spines["bottom"].set_visible(False)
        self.ax1.spines["left"].set_visible(False)
        self.ax1.set_xticks([])
        self.ax1.set_yticks([])
        
        for patch in self.arena.patches:
            if patch.shape == "circle" and self.arena.draw_patches:
                patch_circle = plt.Circle(
                    patch.position,
                    patch.dimensions[0],
                    color="lightgreen",
                    alpha=0.2,
                )

                # Clip to circular arena boundary if needed
                # Create a clipping path using the arena circle
                arena_clip_circle = plt.Circle(
                    self.arena.center,
                    self.arena.arena_radius,
                    transform=self.ax1.transData,
                )
                patch_circle.set_clip_path(arena_clip_circle)

                self.ax1.add_patch(patch_circle)
            elif patch.shape == "rectangle" and self.arena.draw_patches:
                patch_rectangle = plt.Rectangle(
                    patch.position - patch.dimensions / 2,
                    patch.dimensions[0],
                    patch.dimensions[1],
                    color="lightgreen",
                    alpha=0.2,
                )

                # Clip to circular arena boundary if needed
                # Create a clipping path using the arena circle
                arena_clip_circle = plt.Circle(
                    self.arena.center,
                    self.arena.arena_radius,
                    transform=self.ax1.transData,
                )
                patch_circle.set_clip_path(arena_clip_circle)

                self.ax1.add_patch(patch_rectangle)

        for food_pos in self.arena.food_positions:
            self.ax1.plot(
                food_pos[0],
                food_pos[1],
                "go",
                markersize=3,
                label="Food" if food_pos is self.arena.food_positions[0] else "",
            )

        self._plot_projected_sensing(self.ax1)

        # Draw walkerbots (trail + body)
        for idx, wb in enumerate(self.walkerbots):
            # if len(wb.trajectory) > 1:
            #     traj = np.array(wb.trajectory)
            #     self.ax1.plot(traj[:, 0], traj[:, 1], "-", linewidth=1, alpha=0.6, label=None, color="gray")
            if wb.pursuing:
                color = "red"
            else:
                color = "black"
            self.ax1.plot(
                wb.position[0], wb.position[1], marker="o", markersize=7, color=color
            )
            self.ax1.text(
                wb.position[0],
                wb.position[1],
                f"W{idx}",
                ha="center",
                va="center",
                fontsize=5,
                color="white",
                path_effects=[path_effects.withStroke(linewidth=1, foreground="black")],
            )

        for agent in self.agent_objects:
            trajectory = np.array(agent.trajectory)
            if len(trajectory) > 0:
                self.ax1.plot(trajectory[:, 0], trajectory[:, 1], "b-", label=None)
                color = "lightblue"
                # if agent.energy <= 0:
                #     color = 'red'
                self.ax1.plot(
                    agent.position[0],
                    agent.position[1],
                    "o",
                    color=color,
                    markersize=7,
                    label="Agent",
                )
                self.ax1.text(
                    agent.position[0],
                    agent.position[1],
                    str(agent.agent_id),
                    ha="center",
                    va="center",
                    fontsize=5,
                    color="white",
                    path_effects=[
                        path_effects.withStroke(linewidth=1, foreground="black")
                    ],
                )
                # eating_circle = plt.Circle(agent.position, radius=agent.eating_radius, edgecolor='black', facecolor='none')
                # self.ax1.add_patch(eating_circle)

                # # Color arc of eating circle to indicate agent orientation
                # arc_start = np.degrees(agent.orientation - np.pi / 12)
                # arc_end = np.degrees(agent.orientation + np.pi / 12)
                # arc = Arc(agent.position, 2*agent.eating_radius, 2*agent.eating_radius,
                #     angle=np.degrees(agent.orientation), theta1=np.degrees(arc_start),
                #     theta2=np.degrees(arc_end), color='pink', linewidth=1,)
                # self.ax.add_patch(arc)

            # Add overlap region with higher alpha
            # left_eye_orientation = agent.left_eye_angle + agent.orientation
            # right_eye_orientation = agent.right_eye_angle + agent.orientation

            # left_start = left_eye_orientation - agent.perception_field/2
            # left_end = left_eye_orientation + agent.perception_field/2
            # right_start = right_eye_orientation - agent.perception_field/2
            # right_end = right_eye_orientation + agent.perception_field/2

            # # Check for overlap
            # overlap_start = max(left_start, right_start)
            # overlap_end = min(left_end, right_end)

            # if overlap_start < overlap_end:  # There is overlap
            #     overlap_wedge = Wedge(
            #         center=agent.position,
            #         r=agent.ray_length,
            #         theta1=np.degrees(overlap_start),
            #         theta2=np.degrees(overlap_end),
            #         facecolor='yellow',
            #         alpha=0.2,
            #         edgecolor='none'
            #     )
            #     self.ax1.add_patch(overlap_wedge)

            left_eye_pos, right_eye_pos = (
                agent._get_eye_positions()
            )

            for eye in ["left", "right"]:

                eye_pos = left_eye_pos if eye == "left" else right_eye_pos

                eye_orientation = agent.orientation + (
                    agent.left_eye_angle if eye == "left" else agent.right_eye_angle
                )

                start_angle = eye_orientation - agent.perception_field / 2
                end_angle = eye_orientation + agent.perception_field / 2

                # wedge = Wedge(
                #     center=eye_pos,
                #     r=agent.ray_length,
                #     theta1=np.degrees(start_angle),
                #     theta2=np.degrees(end_angle),
                #     facecolor="yellow",
                #     alpha=0.05,
                #     edgecolor="black",
                # )

                # Add smaller wedge for food sensing radius
                food_wedge = Wedge(
                    center=eye_pos,
                    r=agent.food_sensing_radius,
                    theta1=np.degrees(start_angle),
                    theta2=np.degrees(end_angle),
                    facecolor="lightgray",
                    alpha=0.5,
                    edgecolor="lightgray",
                    linewidth=0.5,
                )

                #wedge for fish sensing radius
                fish_wedge = Wedge(
                    center=eye_pos,
                    r=agent.fish_sensing_radius,
                    theta1=np.degrees(start_angle),
                    theta2=np.degrees(end_angle),
                    facecolor="deepskyblue",
                    alpha=0.05,
                    edgecolor="deepskyblue",  # or another color you like
                    linewidth=0.5,
                    linestyle="--",
                )

                # Create a clipping path using the arena circle
                arena_clip_circle = plt.Circle(
                    self.arena.center,
                    self.arena.arena_radius,
                    transform=self.ax1.transData,
                )
                #wedge.set_clip_path(arena_clip_circle)
                food_wedge.set_clip_path(arena_clip_circle)
                fish_wedge.set_clip_path(arena_clip_circle)

                #self.ax1.add_patch(wedge)
                self.ax1.add_patch(food_wedge)
                self.ax1.add_patch(fish_wedge)
    
                # cone_start = agent.position + agent.ray_length * np.array([np.cos(start_angle), np.sin(start_angle)])
                # cone_end = agent.position + agent.ray_length * np.array([np.cos(end_angle), np.sin(end_angle)])

                # self.ax.plot([agent.position[0], cone_start[0]], [agent.position[1], cone_start[1]], 'purple', linewidth=1, alpha=0.5)
                # self.ax.plot([agent.position[0], cone_end[0]], [agent.position[1], cone_end[1]], 'purple', linewidth=1, alpha=0.5)

            # Plot a line facing forward (purple) to show the agent orientation
            # forward_direction = agent.position + agent.ray_length * np.array([np.cos(agent.orientation), np.sin(agent.orientation)])
            # self.ax1.plot([agent.position[0], forward_direction[0]], [agent.position[1], forward_direction[1]], 'purple', linewidth=1, alpha=0.3)

        # self.ax.legend(loc="upper right")
        # self.ax1.legend(loc="upper left", bbox_to_anchor=(1, 1))
        # self.ax1.set_xlim(0 - buffer_area, self.arena_size[0] + buffer_area)
        # self.ax1.set_ylim(0 - buffer_area, self.arena_size[1] + buffer_area)

        aux_index = 0
        for aux_name in aux_plot_funcs:
            ax = self.aux_axes[aux_index]
            aux_plot_funcs[aux_name](ax)
            aux_index += 1

        if mode == "rgb_array":
            buf = io.BytesIO()
            self.fig.savefig(buf, format="png")
            buf.seek(0)
            image = Image.open(buf)
            frame = np.array(image)
            self.frames.append(frame)
            buf.close()
            return frame
        # if mode == "rgb_array":
        #     # Faster alternative suggested by Copilot
        #     canvas = self.fig.canvas
        #     canvas.draw()
        #     w, h = canvas.get_width_height()
        #     buf = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
        #     frame = buf.reshape((h, w, 3)).copy()
        #     return frame


    def _plot_local_view(self, ax):
        """Plot a local view around the first agent, showing what it can perceive."""
        if not self.agent_objects:
            return

        # Focus on the first agent
        agent = self.agent_objects[0]

        # Set the local view size based on the agent's sensing radius
        view_radius = (
            agent.food_sensing_radius * 1.5
        )  # Show a bit more than sensing radius

        ax.clear()
        ax.set_title(f"Local View - Agent {agent.agent_id}")
        ax.set_aspect("equal", "box")

        # Set the view limits centered on the agent
        ax.set_xlim(agent.position[0] - view_radius, agent.position[0] + view_radius)
        ax.set_ylim(agent.position[1] - view_radius, agent.position[1] + view_radius)

        # Draw arena boundaries based on shape
        # Draw circular arena boundary if it's within view
        arena_center = self.arena.center
        arena_radius = self.arena.arena_radius
        agent_to_center_dist = np.linalg.norm(agent.position - arena_center)

        # Check if circle intersects with view area
        if agent_to_center_dist <= view_radius + arena_radius:
            arena_circle = plt.Circle(
                arena_center,
                arena_radius,
                edgecolor="brown",
                fill=False,
                linewidth=2,
                alpha=0.7,
                label="Arena boundary",
            )
            ax.add_patch(arena_circle)

        
        # Draw patches within view
        if self.arena.draw_patches:
            for patch in self.arena.patches:
                if patch.shape == "circle":
                    # Check if patch is within view
                    patch_dist = np.linalg.norm(patch.position - agent.position)
                    if patch_dist <= view_radius + patch.dimensions[0]:
                        patch_circle = plt.Circle(
                            patch.position,
                            patch.dimensions[0],
                            color="lightgreen",
                            alpha=0.2,
                        )
                        # Clip to arena boundary if circular arena
                        arena_clip_circle = plt.Circle(
                            self.arena.center,
                            self.arena.arena_radius,
                            transform=ax.transData,
                        )
                        patch_circle.set_clip_path(arena_clip_circle)
                        ax.add_patch(patch_circle)

        # Draw food within view
        if self.arena.food_positions.size > 0:
            food_distances = np.linalg.norm(
                self.arena.food_positions - agent.position, axis=1
            )
            nearby_food = self.arena.food_positions[food_distances <= view_radius]
            for food_pos in nearby_food:
                ax.plot(food_pos[0], food_pos[1], "go", markersize=3)

        # Draw other agents within view
        for other_agent in self.agent_objects:
            if other_agent != agent:
                other_dist = np.linalg.norm(other_agent.position - agent.position)
                if other_dist <= view_radius:
                    ax.plot(
                        other_agent.position[0],
                        other_agent.position[1],
                        "o",
                        color="lightcoral",
                        markersize=6,
                    )
                    ax.text(
                        other_agent.position[0],
                        other_agent.position[1],
                        str(other_agent.agent_id),
                        ha="center",
                        va="center",
                        fontsize=4,
                        color="white",
                        path_effects=[
                            path_effects.withStroke(linewidth=1, foreground="black")
                        ],
                    )

        # Draw the focal agent
        ax.plot(
            agent.position[0], agent.position[1], "o", color="lightblue", markersize=8
        )
        ax.text(
            agent.position[0],
            agent.position[1],
            str(agent.agent_id),
            ha="center",
            va="center",
            fontsize=6,
            color="white",
            path_effects=[path_effects.withStroke(linewidth=1, foreground="black")],
        )

        # Draw sensing radius circles
        sensing_circle = plt.Circle(
            agent.position,
            agent.fish_sensing_radius,
            fill=False,
            edgecolor="blue",
            linestyle="--",
            alpha=0.5,
        )
        ax.add_patch(sensing_circle)

        food_sensing_circle = plt.Circle(
            agent.position,
            agent.food_sensing_radius,
            fill=False,
            edgecolor="green",
            linestyle=":",
            alpha=0.5,
        )
        ax.add_patch(food_sensing_circle)

        # Draw perception cones
        left_eye_pos, right_eye_pos = (
            agent._get_eye_positions()
        )

        for eye in ["left", "right"]:
            eye_pos = left_eye_pos if eye == "left" else right_eye_pos
            eye_orientation = agent.orientation + (
                agent.left_eye_angle if eye == "left" else agent.right_eye_angle
            )

            start_angle = eye_orientation - agent.perception_field / 2
            end_angle = eye_orientation + agent.perception_field / 2

            # Perception cone
            wedge = Wedge(
                center=eye_pos,
                r=agent.ray_length,
                theta1=np.degrees(start_angle),
                theta2=np.degrees(end_angle),
                facecolor="lightgray",
                alpha=0.5,
                edgecolor="orange",
            )

            # Clip to arena boundary if circular arena
            arena_clip_circle = plt.Circle(
                self.arena.center, self.arena.arena_radius, transform=ax.transData
            )
            wedge.set_clip_path(arena_clip_circle)

            ax.add_patch(wedge)

        # Draw projected sensing if applicable
        self._plot_projected_sensing(ax)

        ax.grid(True, alpha=0.3)

    def close(self):
        if self.frames:
            # current_time = datetime.now()
            # timestamp = current_time.strftime('%Y%m%d')
            random_string = "".join(random.choices(string.digits, k=8))
            if self.expt_name:
                random_string = f"{self.expt_name}_{random_string}"
            OUTPUT_FILENAME = (
                f"{self.video_save_dir}/MAZFish_{self.timestamp}_{random_string}.mp4"
            )
            print(f"Saving video to {OUTPUT_FILENAME}")
            imageio.mimsave(
                OUTPUT_FILENAME, self.frames, fps=ENV_PARAMS["fps_video"]
            )


class TestTurnsAgent:
    def __init__(self, max_turn):
        self.max_turn = max_turn
        self.state = "searching"  # "searching", "forward_swim", "turning"
        self.state_counter = 0
        self.turn_direction = 1
        self.time = 0

    def get_action(self):
        # State transitions
        if self.state == "searching":
            if self.state_counter >= 15:  # Search for 15 steps
                self.state = "forward_swim"
                self.state_counter = 0
        elif self.state == "forward_swim":
            if self.state_counter >= 8:  # Swim forward for 8 steps
                self.state = "turning"
                self.state_counter = 0
                self.turn_direction *= -1  # Alternate turning direction
        elif self.state == "turning":
            if self.state_counter >= 5:  # Turn for 5 steps
                self.state = "searching"
                self.state_counter = 0

        self.state_counter += 1
        self.time += 1

        # Action based on current state
        if self.state == "searching":
            # Move slowly, eyes turned outward for wide search
            move_forward = 5
            turn_angle = 0  # forward
            left_eye_turn = -np.inf  # Eyes outward
            right_eye_turn = 0.8
        elif self.state == "forward_swim":
            # Swim fast forward with eyes turned inward for focus
            move_forward = np.inf
            turn_angle = 0
            left_eye_turn = np.inf  # Eyes inward/converged
            right_eye_turn = -0.6
        else:  # turning
            # Turn in place with moderate eye positioning
            move_forward = 0
            turn_angle = np.random.choice([1, 2])
            left_eye_turn = 0.2
            right_eye_turn = -0.2

        discrete_actions = move_forward * 3 + turn_angle

        left_eye_state = self.time % 2

        return np.array([left_eye_turn, move_forward, turn_angle])


def main(all_args, seed=None, NUM_STEPS=100):
    env = MultiAgentZFishEnv(all_args, seed=seed)

    hardcoded_agents = [TestTurnsAgent(max_turn=1) for _ in range(env.num_agents)]

    for _ in tqdm.tqdm(range(NUM_STEPS)):
        actions = np.array([agent.get_action() for agent in hardcoded_agents])

        observations, rewards, terminations, infos = env.step(actions)
        env.render(mode=all_args["render_mode"])
        if np.all(terminations):
            break
    env.close()


if __name__ == "__main__":
    all_args = {
        "num_agents": 1,
        "arena_size": (100, 100),
        "reset_food_density": cfg.ENV_PARAMS["reset_food_density"],  # 0.015,
        "step_food_density": cfg.ENV_PARAMS["step_food_density"],  # 0.00001,
        "step_food_decay": cfg.ENV_PARAMS["step_food_decay"],  # 0.00001,
        "reset_patch_density": cfg.ENV_PARAMS["reset_patch_density"],  # 0.00001,
        "step_patch_density": cfg.ENV_PARAMS["step_patch_density"],  # 0.00001,
        "max_food_density": cfg.ENV_PARAMS["max_food_density"],  # 0.02, #0.02
        "stockpile_density": cfg.ENV_PARAMS["stockpile_density"],
        "uniform_max_food_density": cfg.ENV_PARAMS["uniform_max_food_density"],
        "uniform_reset_food_density": cfg.ENV_PARAMS["uniform_reset_food_density"],
        "uniform_step_food_density": cfg.ENV_PARAMS["uniform_step_food_density"],
        "patchy_max_food_density": cfg.ENV_PARAMS["patchy_max_food_density"],
        "patchy_reset_food_density": cfg.ENV_PARAMS["patchy_reset_food_density"],
        "feeder_max_food_density": cfg.ENV_PARAMS["feeder_max_food_density"],
        "feeder_reset_food_density": cfg.ENV_PARAMS["feeder_reset_food_density"],
        "energy_food": cfg.AGENT_PARAMS["energy_food"],
        # "render_mode": None,
        "render_mode": "rgb_array",
        "sensing_radius": 20,
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "max_food_eaten_per_step": 1,
        "max_episode_length": 1200,
        "pfeeder": 0,
        "prandom": 0,
        "urandom": 1,
        "p_init_closeby": 0.0,
        "is_eval": False,
        "discrete_actions": False,
        "food_detection_range": cfg.AGENT_PARAMS["food_detection_range"],
        "eating_angle": cfg.AGENT_PARAMS["eating_angle"],
        "eating_distribution_decay_start": cfg.AGENT_PARAMS[
            "eating_distribution_decay_start"
        ],
        "eating_distribution_decay_step": cfg.AGENT_PARAMS[
            "eating_distribution_decay_step"
        ],
        "use_1dof_eyes": True,
        "binocular_depth_only": True,
        "binocular_angle_only": True,
        "binary_eye_state": False,
        "eye_persistence": 1,
        "blur_effect": False,
        "vergence_deviation": 0.0,
        "flash_monocular_only": False,
        "eye_muscle_model": False,  # Whether to use eye muscle model for vergence control
        "k_relax_eye": 1,  # Relaxation constant for eye
        "g_input_eye": 2,  # Gain for eye input
        "detection_failure_rate": 0.0,
        "false_positive_rate": 0.0,
        "use_eye_fatigue": True,
        "eye_fatigue_recovery": 2
        * (
            cfg.FISH_CONSTANTS["min_right_vergence"]
            - cfg.FISH_CONSTANTS["max_right_vergence"]
        )
        / 20,
        "eating_distribution_decay_max": cfg.AGENT_PARAMS[
            "eating_distribution_decay_max"
        ],
        "food_speed_max": cfg.ENV_PARAMS["food_speed"],
        "food_speed_step": 0,
        "food_turn_std_max": cfg.ENV_PARAMS["food_turn_std"],
        "food_turn_std_step": 0,
        "num_walkerbots": 10,
        "walkerbot_max_step": 1.0,
        "walkerbot_max_turn": 0.3,
        "conspecific_monocular_perception": True,
        "num_env_steps": 10000, # Dummy for self.total_num_episodes calculation
        "n_rollout_threads": 1, # Dummy for self.total_num_episodes calculation
        "baseline_success_eating_angle": 10 * np.pi / 180,
        "eye_fatigue_penalty": 0.0,
        "angle_noise_std_food": 0.0,
        "angle_noise_std_walker": 0.0,
        "walker_pursuit": True,
        "walker_ignore_walls": True,
    }

    seed = 1
    np.random.seed(seed)
    random.seed(seed)
    main(all_args, seed=seed, NUM_STEPS=20)
