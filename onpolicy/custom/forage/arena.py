# To generate the figures in the manuscript, run the following command:
# python arena.py # for uniform, patchy, and feeder arenas
# python arena.py --patches_per_edge=10 --reset_food_density=10 --stockpile_density=1.0 # for lattice
# TODO: Aaron, WTF

import io
import argparse
import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

from PIL import Image
import imageio

import tqdm

from cfg import ENV_PARAMS


class FoodPellet:
    pellets_produced = 0

    def __init__(self, position, parent):
        self.position = position
        self.velocity = np.zeros_like(self.position)
        self.parent = parent
        self.global_index = self.__class__.pellets_produced
        self.__class__.pellets_produced += 1

    def __hash__(self):
        return hash(self.global_index)

    def remove(self):
        self.parent.remove_food_pellet(self)

    def step(self): #CURRENTLY USING REFLECTION, not bounded walk. TODO: implement in FoodPatch for efficiency.
        heading = np.arctan2(self.velocity[1], self.velocity[0])
        if self.velocity[0] == 0 and self.velocity[1] == 0:
            heading = self.parent.np_random.uniform(0, 2*np.pi)

        #Create new heading
        heading += self.parent.np_random.normal(0, self.parent.food_turn_std)

        #compute velocities
        self.velocity = np.array([
            np.cos(heading),
            np.sin(heading)
        ]) * self.parent.food_speed

        # Reflect off walls

        #Check within arena
        center = np.array([self.parent.arena_radius, self.parent.arena_radius])
        distance_from_center = np.linalg.norm(self.position - center)
        if distance_from_center > self.parent.arena_radius:
            # Reflect off circular boundary
            direction = (self.position - center) / distance_from_center
            self.position = center + direction * self.parent.arena_radius
            
            # Reflect velocity (bounce off wall)
            normal = direction  # Normal vector pointing outward
            self.velocity = self.velocity - 2 * np.dot(self.velocity, normal) * normal


        #Check within food patch
        if self.parent.shape == "circle":
            center = self.parent.position
            distance_from_center = np.linalg.norm(self.position - center)
            if distance_from_center > self.parent.dimensions[0]:
                # Reflect off circular boundary
                direction = (self.position - center) / distance_from_center
                self.position = center + direction * self.parent.dimensions[0]
                
                # Reflect velocity (bounce off wall)
                normal = direction  # Normal vector pointing outward
                self.velocity = self.velocity - 2 * np.dot(self.velocity, normal) * normal
                
                # Update heading to match new velocity direction
                #self.heading = np.arctan2(self.velocity[1], self.velocity[0])
        else: #Rectangular patch
            for ax in (0, 1):
                low = self.position[ax] < self.parent.position[ax]
                if low:
                    self.position[ax] = 2 * (self.parent.position[ax]) - self.position[ax]
                    self.velocity[ax] = -self.velocity[ax]
                    #heading = np.arctan2(self.velocity[1], self.velocity[0])
                    
                high = self.position[ax] > self.parent.position[ax] + self.parent.dimensions[ax]
                if high:
                    self.position[ax] = 2 * (self.parent.position[ax] + self.parent.dimensions[ax]) - self.position[ax]
                    self.velocity[ax] = -self.velocity[ax]
                    #heading = np.arctan2(self.velocity[1], self.velocity[0])


        # Update velocity and position
        #self.velocity = np.array([np.cos(heading), np.sin(heading)]) * self.parent.food_speed
        self.position += self.velocity

class FoodPatch:
    def __init__(
        self,
        parent,
        position,
        shape,
        dimensions,
        arena_size,
        arena_radius,
        reset_food_density,
        step_food_density,
        step_food_decay,
        food_speed,
        food_turn_std,
        max_food_density=float("inf"),
        stockpile_density=float("inf"),
    ):
        self.parent = parent
        self.position = position
        self.shape = shape
        self.dimensions = dimensions
        self.arena_size = arena_size
        self.arena_radius = arena_radius
        self.reset_food_density = reset_food_density
        self.step_food_density = step_food_density
        self.step_food_decay = step_food_decay
        self.max_food_density = max_food_density
        self.stockpile_density = stockpile_density
        self.food_pellets = set()
        self.food_speed = food_speed
        self.food_turn_std = food_turn_std

    def sample_positions(self, n):
        if self.shape == "circle":
            r = self.dimensions[0]
            positions = self.np_random.uniform(low=[-r, -r], high=[r, r], size=(n, 2))

            # resample points outside the circle
            n2 = positions[:, 0] ** 2 + positions[:, 1] ** 2
            noncircles = n2 > r**2
            num_noncircles = np.sum(noncircles)
            if num_noncircles:
                resampled_positions = self.sample_positions(num_noncircles)
                positions[noncircles] = resampled_positions
            positions[~noncircles] = positions[~noncircles] + self.position

        elif self.shape == "rectangle":
            positions = (
                self.np_random.uniform(
                    low=[self.dimensions[0], self.dimensions[1]],
                    high=[self.dimensions[2], self.dimensions[3]],
                    size=(n, 2),
                )
                + self.position
            )
        else:
            raise NotImplementedError

        # Check arena boundaries based on arena shape
        # For circular arena - resample points outside the circle
        arena_radius = getattr(self, 'arena_radius', min(self.arena_size) / 2)
        center = np.array([arena_radius, arena_radius])
        distances = np.linalg.norm(positions - center, axis=1)
        oob = distances > arena_radius
        
        oob_indices = np.where(oob)[0]
        n_oob = oob_indices.shape[0]

        if n_oob:
            resampled_positions = self.sample_positions(n_oob)
            positions[oob_indices] = resampled_positions

        return positions

    @property
    def area(self):
        if self.shape == "circle":
            return np.pi * self.dimensions[0] ** 2
        elif self.shape == "rectangle":
            a = self.dimensions[2] - self.dimensions[0]
            b = self.dimensions[3] - self.dimensions[1]
            return a * b

    @property
    def max_food_in_area(self):
        return int(round(self.max_food_density * self.arena_radius**2 * np.pi * (self.area/self.parent.total_patch_area)))

    def __len__(self):
        return len(self.food_pellets)

    def grow_food_pellets(self, n):
        n = min(n, self.max_food_in_area - len(self), self.stockpile)
        positions = self.sample_positions(np.maximum(n, 0))
        for position in positions:
            self.food_pellets.add(
                FoodPellet(position, self)
            )

        self.stockpile -= n

    def remove_food_pellet(self, food_pellet):
        self.food_pellets.remove(food_pellet)

    def reset(self, seed=None):

        # setup seeded random number generator
        if seed is not None:
            self.np_random = np.random.default_rng(seed=seed)
        elif not hasattr(self, "np_random"):
            self.np_random = np.random.default_rng()

        self.food_pellets = set()
        a = self.arena_radius**2 * np.pi
        self.stockpile = int(round(self.stockpile_density * a * (self.area/self.parent.total_patch_area)))
        n = self.np_random.poisson(self.reset_food_density * a * (self.area/self.parent.total_patch_area))
        self.grow_food_pellets(n)

    def list_food_pellets(self):
        return list(self.food_pellets)

    def step(self):
        n_grow = self.np_random.poisson(self.step_food_density * self.arena_radius**2 * np.pi * (self.area/self.parent.total_patch_area))
        self.grow_food_pellets(n_grow)

        decay = self.np_random.random(len(self)) < self.step_food_decay
        decay_indices = np.where(decay)[0]
        pellet_list = self.list_food_pellets()
        for i in decay_indices:
            self.remove_food_pellet(pellet_list[i])

        for pellet in pellet_list:
           pellet.step()


class Arena:
    def __init__(
        self,
        min_arena_size,
        max_arena_size,
        reset_food_density=0.01,
        step_food_density=0.001,
        step_food_decay=0.0,
        max_food_density=0.05,
        stockpile_density=1.0,
        food_speed=ENV_PARAMS["food_speed"],
        food_turn_std=ENV_PARAMS["food_turn_std"],
    ):
        self.min_arena_size = min_arena_size
        self.max_arena_size = max_arena_size
        self.reset_food_density = reset_food_density
        self.step_food_density = step_food_density
        self.step_food_decay = step_food_decay
        self.max_food_density = max_food_density
        self.stockpile_density = stockpile_density
        self.food_radius = ENV_PARAMS["food_radius"]
        self.draw_patches = True
        self.food_speed = food_speed
        self.food_turn_std = food_turn_std

    @property
    def area(self):
        return np.pi * self.arena_radius ** 2
        
    @property
    def num_patches(self):
        return len(self.patches)

    @property
    def food_pellets(self):
        return sum((patch.list_food_pellets() for patch in self.patches), [])

    @property
    def num_food_pellets(self):
        return len(self.food_pellets)

    @property
    def food_positions(self):
        if len(self.food_pellets):
            return np.stack([pellet.position for pellet in self.food_pellets])
        else:
            return np.zeros((0, 2))

    @property
    def food_velocities(self):
        if len(self.food_pellets):
            return np.stack([pellet.velocity for pellet in self.food_pellets])
        else:
            return np.zeros((0, 2))
    
    @property
    def center(self):
        return np.array([self.arena_radius, self.arena_radius])
    
    @property
    def total_patch_area(self):
        if not hasattr(self, "patches") or not len(self.patches):
            return 0.0
        return sum(patch.area for patch in self.patches)
        
    def is_position_valid(self, position):
        """Check if a position is within the arena."""
        center = self.center
        distance_from_center = np.linalg.norm(position - center)
        return distance_from_center <= self.arena_radius

    def remove_food_pellet(self, index):
        self.food_pellets[index].remove()

    def eat_food(self, *indices):
        for index in sorted(indices, reverse=True):
            try:
                self.remove_food_pellet(index)
            except:
                breakpoint()

    def prune_unused_patches(self):
        self.patches = [
            patch for patch in self.patches if len(patch) # or patch.stockpile
        ]

    def reset(self, seed=None):
        # setup seeded random number generator
        if seed is not None:
            self.np_random = np.random.default_rng(seed=seed)
        elif not hasattr(self, "np_random"):
            self.np_random = np.random.default_rng()

        # For circular arena, use the minimum dimension as diameter
        min_size = min(self.min_arena_size)
        max_size = min(self.max_arena_size)
        diameter = self.np_random.uniform(low=min_size, high=max_size)
        self.arena_radius = diameter / 2
        self.arena_size = (diameter, diameter)
            
        self.reset_num_patches()
        for patch in self.patches:
            patch_seed = self.np_random.integers(10e8)
            patch.reset(seed=patch_seed)
        self.prune_unused_patches()

    def step(self):
        self.step_num_patches()
        self.prune_unused_patches()
        for patch in self.patches:
            patch.step()

        pos = self.food_positions
        vel = self.food_velocities
        N = pos.shape[0]

        # # initialize headings on first call or if N changed
        # if not hasattr(self, "food_headings") or self.food_headings.shape[0] != N:
        #     headings = np.arctan2(vel[:,1], vel[:,0])
        #     zero_mask = (vel[:,0] == 0) & (vel[:,1] == 0)
        #     if zero_mask.any():
        #         headings[zero_mask] = self.np_random.uniform(0, 2*np.pi, size=zero_mask.sum())
        #     self.food_headings = headings

        # # 1) small random turn
        # self.food_headings += self.np_random.normal(
        #     loc=0,
        #     scale=self.food_turn_std,
        #     size=N
        # )

        # # 2) update velocities
        # speed = self.food_speed
        # vel = np.stack([
        #     np.cos(self.food_headings),
        #     np.sin(self.food_headings)
        # ], axis=1) * speed

        # # 3) step positions
        # pos = pos + vel

        # # 4) handle boundaries based on arena shape
        # # Circular boundary reflection
        # center = self.center
        # for i in range(N):
        #     distance_from_center = np.linalg.norm(pos[i] - center)
        #     if distance_from_center > self.arena_radius:
        #         # Reflect off circular boundary
        #         direction = (pos[i] - center) / distance_from_center
        #         pos[i] = center + direction * self.arena_radius
                
        #         # Reflect velocity (bounce off wall)
        #         normal = direction  # Normal vector pointing outward
        #         vel[i] = vel[i] - 2 * np.dot(vel[i], normal) * normal
                
        #         # Update heading to match new velocity direction
        #         self.food_headings[i] = np.arctan2(vel[i, 1], vel[i, 0])
        
        # # # 4) reflect at borders
        # # for ax in (0, 1):
        # #     low = pos[:,ax] < 0
        # #     if low.any():
        # #         pos[low,ax] = -pos[low,ax]
        # #         vel[low,ax] = -vel[low,ax]
        # #         # Update heading to match new velocity direction
        # #         self.food_headings[low] = np.arctan2(vel[low,1], vel[low,0])
                
        # #     high = pos[:,ax] > self.arena_size[ax]
        # #     if high.any():
        # #         pos[high,ax] = 2*self.arena_size[ax] - pos[high,ax]
        # #         vel[high,ax] = -vel[high,ax]
        # #         # Update heading to match new velocity direction  
        # #         self.food_headings[high] = np.arctan2(vel[high,1], vel[high,0])

        # # 5) write back into pellets
        # for i, pellet in enumerate(self.food_pellets):
        #     pellet.position = pos[i]
        #     pellet.velocity = vel[i]

    @property
    def patch_kwargs(self):
        return {
            "arena_size": self.arena_size,
            "arena_radius": self.arena_radius,
            "reset_food_density": self.reset_food_density,
            "step_food_density": self.step_food_density,
            "step_food_decay": self.step_food_decay,
            "food_speed": self.food_speed,
            "food_turn_std": self.food_turn_std,
            "max_food_density": self.max_food_density,
            "stockpile_density": self.stockpile_density,
        }
    
    def clip_position_to_arena(self, position):
        """Clip a position to be within the arena."""
        center = self.center
        direction = position - center
        distance = np.linalg.norm(direction)
        
        if distance <= self.arena_radius:
            return position
        else:
            # Project onto the circle boundary
            unit_direction = direction / distance
            return center + unit_direction * self.arena_radius
        
    def render(self, path=None, show=False, **kwargs):
        fig, ax = plt.subplots()
        
        # Circular arena rendering
        buffer = 5
        ax.set_xlim(-buffer, 2*self.arena_radius + buffer)
        ax.set_ylim(-buffer, 2*self.arena_radius + buffer)
        ax.set_aspect("equal")

        # Draw circular arena boundary
        arena_circle = plt.Circle(
            self.center,
            self.arena_radius,
            edgecolor="black",
            fill=False,
            linewidth=2,
            clip_on=False
        )
        ax.add_patch(arena_circle)

        # Remove the rectangular axes frame/spines for circular arenas
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Draw patches (same for both shapes)
        for patch in self.patches:
            if patch.shape == "circle" and self.draw_patches:
                patch_circle = plt.Circle(
                    patch.position,
                    patch.dimensions[0],
                    color="lightgreen",
                    alpha=0.2,
                )
                
                # Clip to circular arena boundary if needed
                # Create a clipping path using the arena circle
                arena_clip_circle = plt.Circle(
                self.center,
                self.arena_radius,
                transform=ax.transData
                )
                patch_circle.set_clip_path(arena_clip_circle)

                ax.add_patch(patch_circle)
            elif patch.shape == "rectangle" and self.draw_patches:
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
                    self.center,
                    self.arena_radius,
                    transform=ax.transData
                )
                patch_circle.set_clip_path(arena_clip_circle)

                ax.add_patch(patch_rectangle)

        # Draw food (same for both shapes)
        ax.plot(
            self.food_positions[:, 0],
            self.food_positions[:, 1],
            "go",
            markersize=2 if not "markersize" in kwargs else kwargs["markersize"],
        )

        if "manuscript" in kwargs and kwargs["manuscript"]:
            plt.xlim(0, 2*self.arena_radius)
            plt.ylim(0, 2*self.arena_radius)
            plt.xticks([])
            plt.yticks([])

        if path:
            plt.savefig(path)
            print(f"Saved arena visualization to {path}")
        if show:
            plt.show()

        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        image = Image.open(buf)
        frame = np.array(image)
        buf.close()
        plt.close()

        return frame

    def reset_num_patches(self):
        self.patches = []

    def step_num_patches(self):
        pass


class PatchyArena(Arena):
    def __init__(
        self,
        *args,
        patch_d_mean,
        patch_d_var,
        reset_patch_density,
        step_patch_density=0.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.patch_d_mean = patch_d_mean
        self.patch_d_var = patch_d_var
        self.reset_patch_density = reset_patch_density
        self.step_patch_density = step_patch_density

    def add_new_patches(self, n):
        n = min(n, ENV_PARAMS["max_patches"] - len(self.patches))
        patch_centers = [patch.position for patch in self.patches]
        patch_radii = [patch.dimensions[0] for patch in self.patches]
        new_patches = []
        for i in range(n):
            # Sample position within circular arena. Prevent patches too close to edge as well as overlapping too much.
            position = None
            patch_radius = None

            while position is None or any(
                np.linalg.norm(position - pc) < (pr + patch_radius)
                for pc, pr in zip(patch_centers, patch_radii)
            ):
                angle = self.np_random.uniform(0, 2*np.pi)
                radius = self.np_random.uniform(0, self.arena_radius)  # Stay away from edges
                position = self.center + radius * np.array([np.cos(angle), np.sin(angle)])
            
                patch_radius = self.np_random.normal(
                    loc=self.patch_d_mean / 2, scale=self.patch_d_var / 2
                )
            new_patches.append(
                FoodPatch(self, position, "circle", [patch_radius], **self.patch_kwargs)
            )
            patch_centers.append(position)
            patch_radii.append(patch_radius)

        self.patches.extend(new_patches)
        return new_patches

    def reset_num_patches(self):
        self.patches = []
        #n = self.np_random.poisson(self.reset_patch_density * self.area)
        n = int(self.reset_patch_density * self.arena_radius**2 * np.pi)
        self.add_new_patches(n)

    def step_num_patches(self):
        n = self.np_random.poisson(self.step_patch_density * self.arena_radius**2 * np.pi)
        new_patches = self.add_new_patches(n)
        for patch in new_patches:
            patch.reset()


class UniformArena(Arena):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.draw_patches = False

    def reset_num_patches(self):
        # Create a single circular patch that fills most of the arena
        self.patches = [
            FoodPatch(self, 
                position=self.center,
                shape="circle", 
                dimensions=[self.arena_radius * 0.99],  # Slightly smaller than arena
                **self.patch_kwargs
            )
        ]
        
    def prune_unused_patches(self):
        pass


class LatticeArena(Arena):
    def __init__(
        self,
        *args,
        patches_per_edge,
        tightness=0.01,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.patches_per_edge = patches_per_edge
        self.tightness = tightness

    def reset_num_patches(self):
        self.patches = []
        patch_width = self.arena_size[0] / self.patches_per_edge
        patch_height = self.arena_size[1] / self.patches_per_edge

        for ix in range(self.patches_per_edge):
            for iy in range(self.patches_per_edge):
                position_x = (ix + 0.5) * patch_width
                position_y = (iy + 0.5) * patch_height
                patch_dimensions = [0, 0, self.tightness, self.tightness]

                self.patches.append(
                    FoodPatch(self,
                        position=[position_x, position_y],
                        shape="rectangle",
                        dimensions=patch_dimensions,
                        **self.patch_kwargs,
                    )
                )

    @property
    def patch_kwargs(self):
        return {
            "arena_size": self.arena_size,
            "arena_radius": getattr(self, 'arena_radius', None),  # For circular arenas
            "reset_food_density": self.reset_food_density / self.tightness**2,
            "step_food_density": self.step_food_density / self.tightness**2,
            "step_food_decay": self.step_food_decay,
            "food_speed": self.food_speed,
            "food_turn_std": self.food_turn_std,
            "max_food_density": 1.0 / self.tightness**2,
            "stockpile_density": self.stockpile_density / self.tightness**2,
        }

    def step_num_patches(self):
        # This could be left empty if no dynamic changes are desired in the patches.
        pass


class FeederArena(Arena):
    def __init__(
        self,
        *args,
        patch_d_mean,
        patch_d_var,
        patches_per_edge,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.patch_d_mean = patch_d_mean
        self.patch_d_var = patch_d_var
        self.patches_per_edge = patches_per_edge

    def prune_unused_patches(self):
        pass

    def reset_num_patches(self):
        self.patches = []
        min_edge_length = min(self.arena_size)
        width, height = self.arena_size

        x_positions = np.linspace(
            0.5 * width / self.patches_per_edge,
            width - 0.5 * width / self.patches_per_edge,
            self.patches_per_edge,
        )
        y_positions = np.linspace(
            0.5 * height / self.patches_per_edge,
            height - 0.5 * height / self.patches_per_edge,
            self.patches_per_edge,
        )

        centers = np.concatenate(
            [
                np.stack([x_positions, np.zeros_like(x_positions)], axis=1),
                np.stack([np.full_like(y_positions, width), y_positions], axis=1),
                np.stack([x_positions, np.full_like(x_positions, height)], axis=1),
                np.stack([np.zeros_like(y_positions), y_positions], axis=1),
            ]
        )

        for center in centers:
            radius = np.clip(
                # self.np_random.standard_normal(loc=self.patch_d_mean / 2, scale=self.patch_d_var / 2),
                self.np_random.standard_normal() * self.patch_d_var / 2.0
                + self.patch_d_mean / 2.0,
                0.1,
                min_edge_length / 2,
            )
            self.patches.append(
                FoodPatch(self,
                    position=center,
                    shape="circle",
                    dimensions=[radius],
                    **self.patch_kwargs,
                )
            )


class OnePatchArena(Arena):
    def __init__(
        self,
        *args,
        patch_d_mean,
        patch_d_var,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.patch_d_mean = patch_d_mean
        self.patch_d_var = patch_d_var

    def reset_num_patches(self):
        center = np.array(self.arena_size) / 2
        radius = self.np_random.normal(
            loc=self.patch_d_mean / 2, scale=self.patch_d_var / 2
        )
        radius = np.clip(radius, 0.1, min(self.arena_size) / 2)
        self.patches = [
            FoodPatch(self,
                position=center,
                shape="circle",
                dimensions=[radius],
                **self.patch_kwargs,
            )
        ]

    def prune_unused_patches(self):
        pass


class OnePelletArena(Arena):
    # TODO: Incomplete (doesn't seem to produce a 1-pellet arena)
    def reset_num_patches(self):
        center = np.array(self.arena_size) / 2
        dummy_patch = FoodPatch(self,
            position=center,
            shape="circle",
            dimensions=[0.1],
            **self.patch_kwargs,
        )
        self.patches = [dummy_patch]
        dummy_patch.np_random = self.np_random
        dummy_patch.stockpile = 1
        dummy_patch.max_food_density = 1.0  # prevent cap based on area
        dummy_patch.food_pellets = set()
        pellet = FoodPellet(position=center.copy(), parent=dummy_patch)
        dummy_patch.food_pellets.add(pellet)
        dummy_patch.stockpile = 0

    def prune_unused_patches(self):
        pass

    def step(self):
        # No movement or replenishment
        pass


class SquareQuadrantArena(Arena):
    def __init__(self, *args,
                 num_radial_patches=4,
                 patch_radius=7.5,
                 center_location="bottom_right",
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.num_radial_patches = num_radial_patches
        self.patch_radius = patch_radius
        self.center_location = center_location

    def reset_num_patches(self):
        self.patches = []
        w, h = self.arena_size

        center_map = {
            "bottom_left": np.array([0.0 + self.patch_radius, 0.0 + self.patch_radius]),
            "bottom_right": np.array([w - self.patch_radius, 0.0 + self.patch_radius]),
            "top_left": np.array([0.0 + self.patch_radius, h - self.patch_radius]),
            "top_right": np.array([w - self.patch_radius, h - self.patch_radius]),
        }
        angle_ranges = {
            "bottom_left": (0.0, np.pi / 2),
            "bottom_right": (np.pi / 2, np.pi),
            "top_left": (-np.pi / 2, 0.0),
            "top_right": (np.pi, 3 * np.pi / 2),
        }

        center = center_map[self.center_location]
        theta_start, theta_end = angle_ranges[self.center_location]
        max_radius = min(w, h) / 1.2  # radius of circular arc

        thetas = np.linspace(theta_start, theta_end, self.num_radial_patches)
        for theta in thetas:
            offset = np.array([np.cos(theta), np.sin(theta)]) * max_radius
            position = center + offset
            position = np.clip(position, 
                               [self.patch_radius, self.patch_radius], 
                               [w - self.patch_radius, h - self.patch_radius])  # ensure inside bounds

            self.patches.append(
                FoodPatch(self,
                    position=position,
                    shape="circle",
                    dimensions=[self.patch_radius],
                    **self.patch_kwargs,
                )
            )


def demo_animate(args, kwargs):
    if args.arena_style == "uniform":
        arena = UniformArena(**kwargs)
    elif args.arena_style == "patchy":
        arena = PatchyArena(
            patch_d_mean=args.patch_d_mean,
            patch_d_var=args.patch_d_var,
            reset_patch_density=args.reset_patch_density,
            step_patch_density=args.step_patch_density,
            **kwargs,
        )
    elif args.arena_style == "feeder":
        arena = FeederArena(
            patch_d_mean=args.patch_d_mean,
            patch_d_var=args.patch_d_var,
            patches_per_edge=args.patches_per_edge,
            **kwargs,
        )
    elif args.arena_style == "onepatch":
        arena = OnePatchArena(
            patch_d_mean=args.patch_d_mean,
            patch_d_var=args.patch_d_var,
            **kwargs,
        )
    elif args.arena_style == "onepellet":
        arena = OnePelletArena(**kwargs)
    elif args.arena_style == "lattice":
        arena = LatticeArena(
            patches_per_edge=args.patches_per_edge,
            **kwargs,
        )
    else:
        raise ValueError(f"Unknown arena style: {args.arena_style}")

    frames = []
    arena.reset()
    for i in tqdm.tqdm(range(args.steps)):
        frames.append(arena.render())
        arena.step()

    frames.append(arena.render())

    filename = "./arena.mp4"
    imageio.mimsave(filename, frames, fps=83)


def demo_figures(args, kwargs):
    kwargs["min_arena_size"] = (30, 30)
    kwargs["max_arena_size"] = (30, 30)
    args.reset_patch_density = 0.01
    args.patch_d_mean = 8
    args.patch_d_var = 1
    args.reset_food_density = 0.2

    for arena_type in [
        "uniform",
        "patchy",
        "feeder",
        "onepatch",
        "onepellet",
        # "lattice",  # Note hacky overrides in this section
        "square_quadrant", 
    ]:
        if arena_type == "patchy":
            arena = PatchyArena(
                patch_d_mean=args.patch_d_mean,
                patch_d_var=args.patch_d_var,
                reset_patch_density=args.reset_patch_density,
                step_patch_density=args.step_patch_density,
                **kwargs,
            )
        if arena_type == "feeder":
            arena = FeederArena(
                patch_d_mean=args.patch_d_mean,
                patch_d_var=args.patch_d_var,
                patches_per_edge=args.patches_per_edge,
                **kwargs,
            )
        if arena_type == "onepatch":
            arena = OnePatchArena(
                patch_d_mean=args.patch_d_mean,
                patch_d_var=args.patch_d_var,
                **kwargs,
            )
        if arena_type == "onepellet":
            arena = OnePelletArena(**kwargs)
        if arena_type == "lattice":
            args.patches_per_edge = 10
            args.reset_food_density = 10
            args.stockpile_density = 1.0
            # print(args)
            arena = LatticeArena(
                patches_per_edge=args.patches_per_edge,
                **kwargs,
            )
        if arena_type == "uniform":
            arena = UniformArena(**kwargs)
        if arena_type == "square_quadrant":
            kwargs["min_arena_size"] = (60, 60)
            kwargs["max_arena_size"] = (60, 60)
            for center_location in [
                "bottom_left",
                "bottom_right",
                "top_left",
                "top_right",
                ]:
                arena = SquareQuadrantArena(
                    num_radial_patches=4,
                    patch_radius=7.5,
                    center_location=center_location,
                    **kwargs,
                )
                arena.reset()
                kwargs_render = {
                    "path": f"./arena_{arena_type}_{center_location}.png",
                    "show": False,
                    "manuscript": True,
                    "markersize": 2,  # default: 2,
                }
                frame = arena.render(**kwargs_render)

        arena.reset()
        kwargs_render = {
            "path": f"./arena_{arena_type}.png",
            # "path": f"./arena_{arena_type}.svg",
            "show": False,
            "manuscript": True,
            "markersize": 4,  # default: 2,
        }
        frame = arena.render(**kwargs_render)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--arena_style", type=str, default="uniform")
    parser.add_argument("--reset_food_density", type=float, default=0.05)
    parser.add_argument("--step_food_density", type=float, default=0.0)
    parser.add_argument("--step_food_decay", type=float, default=0.000)
    parser.add_argument("--max_food_density", type=float, default=0.25)
    parser.add_argument("--stockpile_density", type=float, default=0.5)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--patch_d_mean", type=float, default=10)
    parser.add_argument("--patch_d_var", type=float, default=2)
    parser.add_argument("--reset_patch_density", type=float, default=0.001)
    parser.add_argument("--step_patch_density", type=float, default=0.00000)
    parser.add_argument("--patches_per_edge", type=int, default=1)
    args = parser.parse_args()

    min_arena_size = (60, 60)
    max_arena_size = (60, 60)
    kwargs = {
        "min_arena_size": min_arena_size,
        "max_arena_size": max_arena_size,
        "reset_food_density": args.reset_food_density,
        "step_food_density": args.step_food_density,
        "step_food_decay": args.step_food_decay,
        "max_food_density": args.max_food_density,
        "stockpile_density": args.stockpile_density,
    }

    # demo_animate(args, kwargs)
    demo_figures(args, kwargs)