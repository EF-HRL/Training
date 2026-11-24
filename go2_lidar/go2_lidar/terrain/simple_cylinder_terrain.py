# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Simple flat terrain generator with random cylinder obstacles."""

from __future__ import annotations

import datetime
import os
from typing import TYPE_CHECKING

import numpy as np
import omni.log
import torch
import trimesh
from isaaclab.terrains.trimesh.utils import make_border
from isaaclab.terrains.utils import color_meshes_by_height, find_flat_patches
from isaaclab.utils.dict import dict_to_md5_hash
from isaaclab.utils.io import dump_yaml
from isaaclab.utils.timer import Timer
from isaaclab.utils.warp import convert_to_warp_mesh

if TYPE_CHECKING:
    from isaaclab.terrains.sub_terrain_cfg import FlatPatchSamplingCfg, SubTerrainBaseCfg
    from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg


class SimpleFlatTerrainNavigation:
    """Simple navigation guide for flat terrain with random obstacles.
    
    Provides optimal direction vectors pointing from current positions toward specified goals.
    Unlike MazeGridNavigation which uses computed direction maps, this directly computes
    the direction toward goals on flat terrain.
    """
    
    def __init__(self, goal_positions: np.ndarray, device: str = "cpu"):
        """Initialize the navigation guide.
        
        Args:
            goal_positions: Array of shape (N, 3) containing goal positions [x, y, z].
            device: Device to use for tensors.
        """
        self.device = device
        # Store goal positions as tensor for efficient computation
        if isinstance(goal_positions, np.ndarray):
            self.goal_positions = torch.tensor(goal_positions, dtype=torch.float32, device=device)
        else:
            self.goal_positions = goal_positions
        self.num_goals = self.goal_positions.shape[0]
    
    def get_optimal_direction(self, x, y, goal_index):
        """Get optimal direction toward specified goal.
        
        Args:
            x: Tensor or array of x coordinates, shape (batch,)
            y: Tensor or array of y coordinates, shape (batch,)
            goal_index: Tensor or array of goal indices, shape (batch,)
            
        Returns:
            Tensor of shape (batch, 2): Unit direction vectors [dx, dy] toward goals
        """
        # Convert to tensors if needed
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32, device=self.device)
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y, dtype=torch.float32, device=self.device)
        if not isinstance(goal_index, torch.Tensor):
            goal_index = torch.tensor(goal_index, dtype=torch.long, device=self.device)
        
        # Move to correct device
        x = x.to(self.device)
        y = y.to(self.device)
        goal_index = goal_index.to(self.device)
        
        # Get original shape
        original_shape = x.shape
        
        # Flatten to 1D for processing
        x_flat = x.flatten()
        y_flat = y.flatten()
        goal_index_flat = goal_index.flatten()
        batch_size = x_flat.shape[0]
        
        # Initialize output
        directions = torch.zeros((batch_size, 2), dtype=torch.float32, device=self.device)
        
        # Clamp goal indices to valid range
        goal_index_flat = torch.clamp(goal_index_flat, 0, self.num_goals - 1)
        
        # Compute direction vectors for each position toward its goal
        for i in range(batch_size):
            goal_idx = goal_index_flat[i].item()
            goal_pos = self.goal_positions[goal_idx]
            
            # Current position
            curr_x = x_flat[i]
            curr_y = y_flat[i]
            
            # Direction to goal
            dx = goal_pos[0] - curr_x
            dy = goal_pos[1] - curr_y
            
            # Normalize to unit vector
            dist = torch.sqrt(dx**2 + dy**2)
            if dist > 1e-6:
                directions[i, 0] = dx / dist
                directions[i, 1] = dy / dist
            else:
                # Already at goal
                directions[i] = 0.0
        
        # Reshape to match input shape plus 2 for direction components
        return directions.view(*original_shape, 2)


class SimpleCylinderTerrainGenerator:
    """Terrain generator for flat terrain with random cylinder obstacles.
    
    This generator creates a simple flat terrain and places random cylinder obstacles
    on the surface. It's designed as a lightweight alternative to NavTerrainGenerator
    for navigation training when maze-like procedural terrain is not needed.
    """

    terrain_mesh: trimesh.Trimesh
    """A single trimesh.Trimesh object for all the generated sub-terrains."""
    terrain_meshes: list[trimesh.Trimesh]
    """List of trimesh.Trimesh objects for all the generated sub-terrains."""
    terrain_origins: np.ndarray
    """The origin of each sub-terrain. Shape is (num_rows, num_cols, 3)."""
    flat_patches: dict[str, torch.Tensor]
    """A dictionary of sampled valid (flat) patches for each sub-terrain."""

    def __init__(self, cfg: TerrainGeneratorCfg, device: str = "cpu"):
        """Initialize the terrain generator.

        Args:
            cfg: Configuration for the terrain generator.
            device: The device to use for the flat patches tensor.
        """
        # check inputs
        if len(cfg.sub_terrains) == 0:
            raise ValueError("No sub-terrains specified! Please add at least one sub-terrain.")
        
        # store inputs
        self.cfg = cfg
        self.device = device

        # set common values to all sub-terrains config
        from isaaclab.terrains.height_field import HfTerrainBaseCfg  # prevent circular import

        for sub_cfg in self.cfg.sub_terrains.values():
            sub_cfg.size = self.cfg.size
            if isinstance(sub_cfg, HfTerrainBaseCfg):
                sub_cfg.horizontal_scale = self.cfg.horizontal_scale
                sub_cfg.vertical_scale = self.cfg.vertical_scale
                sub_cfg.slope_threshold = self.cfg.slope_threshold

        # handle seed
        if self.cfg.seed is not None:
            seed = self.cfg.seed
        else:
            seed = np.random.get_state()[1][0]
        self.np_rng = np.random.default_rng(seed)

        # buffer for storing valid patches
        self.flat_patches = {}
        self.terrain_meshes = list()
        self.terrain_origins = np.zeros((self.cfg.num_rows, self.cfg.num_cols, 3))

        # generate terrains
        with Timer("[INFO] Generating flat terrains with cylinders took"):
            self._generate_flat_terrains_with_cylinders()
        
        # add a border around the terrains
        self._add_terrain_border()
        
        # combine all the sub-terrains into a single mesh
        self.terrain_mesh = trimesh.util.concatenate(self.terrain_meshes)

        # color the terrain mesh
        if self.cfg.color_scheme == "height":
            self.terrain_mesh = color_meshes_by_height(self.terrain_mesh)
        elif self.cfg.color_scheme == "random":
            self.terrain_mesh.visual.vertex_colors = self.np_rng.choice(
                range(256), size=(len(self.terrain_mesh.vertices), 4)
            )
        elif self.cfg.color_scheme == "none":
            pass
        else:
            raise ValueError(f"Invalid color scheme: {self.cfg.color_scheme}.")

        # offset the entire terrain and origins so that it is centered
        transform = np.eye(4)
        transform[:2, -1] = -self.cfg.size[0] * self.cfg.num_rows * 0.5, -self.cfg.size[1] * self.cfg.num_cols * 0.5
        self.terrain_mesh.apply_transform(transform)
        self.terrain_origins += transform[:3, -1]
        
        # convert flat patches lists to tensors
        for key in list(self.flat_patches.keys()):
            if isinstance(self.flat_patches[key], list) and len(self.flat_patches[key]) > 0:
                self.flat_patches[key] = torch.tensor(
                    self.flat_patches[key], dtype=torch.float32, device=self.device
                )
        
        # offset flat patches
        for name, value in self.flat_patches.items():
            if not isinstance(value, torch.Tensor):
                continue
            self.flat_patches[name][..., 0] += -self.cfg.size[0] * self.cfg.num_rows * 0.5
            self.flat_patches[name][..., 1] += -self.cfg.size[1] * self.cfg.num_cols * 0.5
        
        # generate required flat patches for navigation environment
        self._generate_navigation_flat_patches()

    def _generate_navigation_flat_patches(self):
        """Generate flat patches required by the navigation environment.
        
        Creates patches for:
        - general_start, general_goal
        - dead_end_start, dead_end_goal  
        - detour_start, detour_goal
        - guide generators for each type
        """
        num_patches = 500  # number of spawn/goal points per category
        
        # Create patches for all three navigation types
        for category in ["general", "dead_end", "detour"]:
            # Sample random start and goal positions on flat terrain
            starts = self._sample_random_patches(num_patches)
            goals = self._sample_random_patches(num_patches)
            
            # Store as tensors
            self.flat_patches[f"{category}_start"] = torch.tensor(
                starts, dtype=torch.float32, device=self.device
            )
            self.flat_patches[f"{category}_goal"] = torch.tensor(
                goals, dtype=torch.float32, device=self.device
            )
            
            # Create a simple navigation guide generator for flat terrain
            self.flat_patches[f"{category}_guide_generator"] = SimpleFlatTerrainNavigation(
                goals, self.device
            )

    def _sample_random_patches(self, num_samples: int) -> np.ndarray:
        """Sample random positions on flat terrain.
        
        Args:
            num_samples: Number of samples to generate.
            
        Returns:
            Array of shape (num_samples, 3) with x, y, z coordinates.
        """
        # Calculate terrain bounds
        total_width = self.cfg.size[0] * self.cfg.num_rows
        total_length = self.cfg.size[1] * self.cfg.num_cols
        
        half_width = total_width * 0.5
        half_length = total_length * 0.5
        
        # Sample random positions within bounds
        x = self.np_rng.uniform(-half_width * 0.9, half_width * 0.9, num_samples)
        y = self.np_rng.uniform(-half_length * 0.9, half_length * 0.9, num_samples)
        z = np.zeros(num_samples, dtype=np.float32)
        
        return np.column_stack([x, y, z])

    def __str__(self):
        """Return a string representation of the terrain generator."""
        msg = "Simple Cylinder Terrain Generator:"
        msg += f"\n\tSeed: {self.cfg.seed}"
        msg += f"\n\tNumber of rows: {self.cfg.num_rows}"
        msg += f"\n\tNumber of columns: {self.cfg.num_cols}"
        msg += f"\n\tSub-terrain size: {self.cfg.size}"
        msg += f"\n\tColor scheme: {self.cfg.color_scheme}"
        return msg

    def _generate_flat_terrains_with_cylinders(self):
        """Generate flat terrain with random cylinder obstacles."""
        from isaaclab.terrains import HfRandomUniformTerrainCfg
        
        # get the sub-terrain configuration
        sub_terrain_cfg = list(self.cfg.sub_terrains.values())[0]
        
        # generate all terrain patches
        for row in range(self.cfg.num_rows):
            for col in range(self.cfg.num_cols):
                # 1. create a flat mesh for this terrain patch
                flat_mesh = self._create_flat_patch()
                
                # 2. add random cylinders on top
                flat_mesh = self._add_random_cylinders(flat_mesh, row, col)
                
                # 计算这个子地形的起始原点
                origin_x = row * self.cfg.size[0]
                origin_y = col * self.cfg.size[1]
                self.terrain_origins[row, col] = [origin_x, origin_y, 0.0]
                
                # 【关键修复】：将这个地形网格物理平移到它在全局大网格中的对应位置！
                flat_mesh.apply_translation([origin_x, origin_y, 0.0])
                
                # 3. 必须在地形平移后，再采样出生点，这样采样点才会有正确的全局坐标
                if hasattr(sub_terrain_cfg, 'flat_patch_sampling') and sub_terrain_cfg.flat_patch_sampling is not None:
                    self._sample_flat_patches(flat_mesh, row, col, sub_terrain_cfg.flat_patch_sampling)
                
                # 4. add mesh to list
                self.terrain_meshes.append(flat_mesh)

    def _create_flat_patch(self):
        """Create a flat mesh patch with slight thickness for stable PhysX collision.
        
        Returns:
            trimesh.Trimesh: A flat rectangular mesh box.
        """
        size_x = self.cfg.size[0]
        size_y = self.cfg.size[1]
        thickness = 0.1  # 给予地面 0.1 米的厚度
        
        # 直接生成一个实体 Box
        mesh = trimesh.creation.box(extents=(size_x, size_y, thickness))
        
        # 默认生成的 box 中心在 (0,0,0)
        # 我们将其平移，使地表处于 z=0，并且起点在 (0,0) 与原代码对齐
        mesh.apply_translation([size_x / 2.0, size_y / 2.0, -thickness / 2.0])
        
        return mesh

    def _add_random_cylinders(self, mesh: trimesh.Trimesh, row_idx: int, col_idx: int):
        """Add random cylinder obstacles to the terrain mesh.
        
        Args:
            mesh: The terrain mesh to add cylinders to.
            row_idx: Row index of the terrain patch.
            col_idx: Column index of the terrain patch.
            
        Returns:
            trimesh.Trimesh: The mesh with cylinders added.
        """
        size_x = self.cfg.size[0]
        size_y = self.cfg.size[1]
        
        # parameters for cylinder placement
        grid_size = 5  # 5x5 grid of cylinders
        cylinder_radius = 0.15  # meters
        base_cylinder_height = self.np_rng.uniform(0.3, 1.2)  # random height between 0.3 and 1.2 meters
        
        # calculate cell size
        cell_x = size_x / grid_size
        cell_y = size_y / grid_size
        
        # list to store all meshes to concatenate
        meshes_to_combine = [mesh]
        
        # try to place cylinders in a grid with randomness
        for i in range(grid_size):
            for j in range(grid_size):
                # random chance to place a cylinder (50% chance)
                if self.np_rng.random() > 0.5:
                    continue
                
                # calculate base position
                base_x = (i + 0.5) * cell_x
                base_y = (j + 0.5) * cell_y
                
                # add random offset
                offset_x = self.np_rng.uniform(-cell_x * 0.3, cell_x * 0.3)
                offset_y = self.np_rng.uniform(-cell_y * 0.3, cell_y * 0.3)
                
                center_x = base_x + offset_x
                center_y = base_y + offset_y
                
                # clamp to terrain bounds
                center_x = np.clip(center_x, cylinder_radius, size_x - cylinder_radius)
                center_y = np.clip(center_y, cylinder_radius, size_y - cylinder_radius)
                
                # vary height slightly
                height = self.np_rng.uniform(max(0.3, base_cylinder_height - 0.3), base_cylinder_height + 0.3)
                
                # create cylinder
                cylinder = trimesh.creation.cylinder(
                    radius=cylinder_radius,
                    height=height,
                    sections=16
                )
                
                # position cylinder
                center_z = height / 2.0
                cylinder.apply_translation([center_x, center_y, center_z])
                
                # add to list
                meshes_to_combine.append(cylinder)
        
        # concatenate all meshes into one
        if len(meshes_to_combine) > 1:
            combined_mesh = trimesh.util.concatenate(meshes_to_combine)
            return combined_mesh
        else:
            return mesh

    def _add_terrain_border(self):
        """Add a border around all terrains.
        
        Note: For flat terrain with cylinders, we skip border creation since
        the terrain is already simple. If borders are needed, they should be
        added at a different stage in the pipeline.
        """
        # Border creation is skipped for simple flat terrain
        # The terrain mesh vertices already extend beyond the usable area,
        # so physical borders are not necessary for flat terrain navigation
        pass

    def _sample_flat_patches(
        self, 
        mesh: trimesh.Trimesh, 
        row_idx: int, 
        col_idx: int,
        flat_patch_cfg: dict,
    ):
        """Sample flat patches on the terrain.
        
        Args:
            mesh: The terrain mesh.
            row_idx: Row index of the terrain patch.
            col_idx: Column index of the terrain patch.
            flat_patch_cfg: Dictionary of flat patch configurations.
        """
        # iterate over all patch sampling configurations
        for patch_name, patch_cfg in flat_patch_cfg.items():
            if patch_name not in self.flat_patches:
                self.flat_patches[patch_name] = []
            
            # for flat terrain, we can sample patches easily
            # find areas suitable for spawning (high areas without obstacles)
            patches = self._find_flat_spawn_points(mesh, patch_cfg)
            
            # store patches with row, col indices
            if patches is not None and len(patches) > 0:
                self.flat_patches[patch_name].extend(patches)
        
        # convert lists to tensors at the end (done in __init__)

    def _find_flat_spawn_points(self, mesh: trimesh.Trimesh, patch_cfg):
        """Find suitable flat spawn points on the mesh.
        
        Args:
            mesh: The terrain mesh.
            patch_cfg: Patch sampling configuration.
            
        Returns:
            List of suitable spawn points.
        """
        try:
            patches = find_flat_patches(
                mesh,
                patch_cfg.patch_radius,
                patch_cfg.max_height_diff,
                patch_cfg.num_patches,
                self.np_rng,
            )
            return patches
        except Exception as e:
            # if flat patch finding fails, return empty list
            omni.log.warn(f"Failed to find flat patches: {e}")
            return []