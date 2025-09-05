# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tactile gripper sorting environment with Elfin robot for soft/hard object classification."""

from __future__ import annotations

import math
import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg, AssetBase, AssetBaseCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import FrameTransformer, FrameTransformerCfg, ContactSensor, ContactSensorCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.sim.spawners.from_files import spawn_ground_plane
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import sample_uniform
from isaaclab.markers.visualization_markers import VisualizationMarkersCfg

import os
pwd = os.getcwd()


@configclass
class ElfinTactileLiftEnvCfg(DirectRLEnvCfg):
    """Configuration for tactile gripper sorting environment with Elfin robot."""
    
    # env
    episode_length_s = 8.3333  # 500 timesteps at 60Hz
    decimation = 2  # 60Hz control frequency
    action_space = 8  # 6 arm joints + 2 gripper
    observation_space = 41  # joint pos (9) + joint vel (9) + ee pose (7) + tactile (5) + objects (6) + target (3) + actions (9)
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,  # 120Hz physics
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )
    sim.physx.bounce_threshold_velocity = 0.2
    sim.physx.bounce_threshold_velocity = 0.01
    sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
    sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
    sim.physx.friction_correlation_distance = 0.00625

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=64, env_spacing=2.5, replicate_physics=True)

    # robot - Elfin
    robot = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{pwd}/../model/elfin-gripper2.usd",
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                disable_gravity=False,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True, 
                solver_position_iteration_count=8, 
                solver_velocity_iteration_count=0
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                "elfin_joint1": 0.0 * math.pi / 180.0, 
                "elfin_joint2": 0.0 * math.pi / 180.0, 
                "elfin_joint3": 97.9 * math.pi / 180.0, 
                "elfin_joint4": 0.0 * math.pi / 180.0, 
                "elfin_joint5": 58.2 * math.pi / 180.0, 
                "elfin_joint6": -90 * math.pi / 180.0,
                "finger1": 0.0,
                "finger2": 0.0,
            },
            pos=(0.0, 0.0, 0.0),
        ),
        actuators={
            "joint_actuator1": ImplicitActuatorCfg(
                joint_names_expr=["elfin_joint1"], 
                effort_limit_sim=87.0,
                stiffness=400, 
                damping=80
            ),
            "joint_actuator2": ImplicitActuatorCfg(
                joint_names_expr=["elfin_joint2"], 
                effort_limit_sim=87.0,
                stiffness=400, 
                damping=80
            ),
            "joint_actuator3": ImplicitActuatorCfg(
                joint_names_expr=["elfin_joint3"], 
                effort_limit_sim=87.0,
                stiffness=400, 
                damping=80
            ),
            "joint_actuator4": ImplicitActuatorCfg(
                joint_names_expr=["elfin_joint4"], 
                effort_limit_sim=12.0,
                stiffness=400, 
                damping=80
            ),
            "joint_actuator5": ImplicitActuatorCfg(
                joint_names_expr=["elfin_joint5"], 
                effort_limit_sim=12.0,
                stiffness=400, 
                damping=80
            ),
            "joint_actuator6": ImplicitActuatorCfg(
                joint_names_expr=["elfin_joint6"], 
                effort_limit_sim=12.0,
                stiffness=400, 
                damping=80
            ),
            "finger1_actuator": ImplicitActuatorCfg(
                joint_names_expr=["finger1"], 
                effort_limit_sim=200.0,
                stiffness=1e6, 
                damping=1e2
            ),
            "finger2_actuator": ImplicitActuatorCfg(
                joint_names_expr=["finger2"], 
                effort_limit_sim=200.0,
                stiffness=1e6, 
                damping=1e2
            ),
        },
        soft_joint_pos_limit_factor=1.0,
    )

    # tactile sensors on gripper fingers
    tactile_sensors = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/Gripper/finger1",
        update_period=0.0,  # update every step
        history_length=6,
        debug_vis=False,
        filter_prim_paths_expr=[
            "/World/envs/env_.*/Robot/Gripper/finger1",
            "/World/envs/env_.*/Robot/Gripper/finger2",
        ]
    )

    # end-effector frame with visualization
    ee_frame = FrameTransformerCfg(
        prim_path="/World/envs/env_.*/Robot/Robot/elfin_base_link",
        debug_vis=False,
        visualizer_cfg=VisualizationMarkersCfg(
            prim_path="/Visuals/FrameTransformer",
            markers={
                "frame": sim_utils.UsdFileCfg(
                    usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
                    scale=(0.1, 0.1, 0.1),
                ),
                "connecting_line": sim_utils.CylinderCfg(
                    radius=0.002,
                    height=1.0,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 0.0), roughness=1.0),
                ),
            }
        ),
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="/World/envs/env_.*/Robot/Gripper/base_link",
                name="end_effector",
                offset=OffsetCfg(
                    pos=(-0.03854, 0.00436, 0.11706),
                ),
            ),
        ],
    )

    # soft object - water bottle (light, bouncy)
    soft_bottle = RigidObjectCfg(
        prim_path="/World/envs/env_.*/SoftBottle",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.1, 0.025), rot=(1.0, 0.0, 0.0, 0.0)),
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned/003_cracker_box.usd",
            scale=(0.8, 0.8, 0.8),
            rigid_props=RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
                mass=0.05,  # Very light - soft object
            ),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.5,
                dynamic_friction=0.5,
                restitution=0.8,  # Bouncy
            ),
        ),
    )

    # hard object - can (heavy, rigid)  
    hard_can = RigidObjectCfg(
        prim_path="/World/envs/env_.*/HardCan",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, -0.1, 0.025), rot=(1.0, 0.0, 0.0, 0.0)),
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned/005_tomato_soup_can.usd",
            scale=(0.8, 0.8, 0.8),
            rigid_props=RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
                mass=0.2,  # Heavy - hard object
            ),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.7,
                dynamic_friction=0.7,
                restitution=0.1,  # Rigid
            ),
        ),
    )

    action_scale = 0.5
    dof_velocity_scale = 0.1

    # reward scales for tactile-based sorting
    tactile_reward_scale = 2.0
    force_control_scale = 3.0
    sorting_accuracy_scale = 5.0
    approach_reward_scale = 1.0
    action_penalty_scale = -0.01
    joint_vel_penalty_scale = -0.001
    drop_penalty_scale = -10.0  # heavy penalty for dropping objects
    stable_grasp_reward_scale = 8.0  # reward for stable grasping

    # task parameters
    object_init_pose_range = {"x": (0.0, 0.2), "y": (-0.25, 0.25), "z": (0.0, 0.0)}
    minimal_height = 0.04
    goal_threshold = 0.15  # distance to sorting areas
    stable_grasp_time = 10.0  # seconds to hold object stably
    friction_threshold = 2.0  # minimum friction force for stable grasp


class ElfinTactileLiftEnv(DirectRLEnv):
    """Environment for tactile gripper sorting with Elfin robot."""

    cfg: ElfinTactileLiftEnvCfg

    def __init__(self, cfg: ElfinTactileLiftEnvCfg, render_mode: str | None = None, **kwargs):
        """Initialize the environment."""
        super().__init__(cfg, render_mode, **kwargs)

        self.dt = self.cfg.sim.dt * self.cfg.decimation

        # create auxiliary variables for computing applied action, observations and rewards
        self.robot_dof_lower_limits = self._robot.data.soft_joint_pos_limits[0, :, 0].to(device=self.device)
        self.robot_dof_upper_limits = self._robot.data.soft_joint_pos_limits[0, :, 1].to(device=self.device)

        self.robot_dof_speed_scales = torch.ones_like(self.robot_dof_lower_limits)
        # Find finger joints for special scaling
        finger_indices = []
        try:
            finger_indices.extend(self._robot.find_joints("elfin_finger_joint.*"))
        except ValueError:
            pass
        
        for idx in finger_indices:
            if 0 <= idx < len(self.robot_dof_speed_scales):
                self.robot_dof_speed_scales[idx] = 0.1

        self.robot_dof_targets = torch.zeros((self.num_envs, self._robot.num_joints), device=self.device)

        # Object classification variables
        self.object_softness = torch.zeros(self.num_envs, device=self.device)  # 0=hard, 1=soft
        self.target_areas = torch.zeros((self.num_envs, 3), device=self.device)  # target sorting areas

        # Grasp stability tracking
        self.stable_grasp_timer = torch.zeros(self.num_envs, device=self.device)  # time object held stably
        self.is_grasping = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)  # currently grasping
        self.grasp_success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)  # successful grasp
        self.object_lifted = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)  # object above table

        # Sample initial target areas for sorting
        self._sample_target_areas()

        # Initialize previous actions for penalty calculation
        self.prev_actions = torch.zeros((self.num_envs, self.cfg.action_space), dtype=torch.float32, device=self.device)

    def _setup_scene(self):
        """Setup the scene entities."""
        spawn_ground_plane("/World/ground", cfg=GroundPlaneCfg())
        self._robot = Articulation(self.cfg.robot)
        self._soft_bottle = RigidObject(self.cfg.soft_bottle)
        self._hard_can = RigidObject(self.cfg.hard_can)
        self._tactile_sensors = ContactSensor(self.cfg.tactile_sensors)
        self._ee_frame = FrameTransformer(self.cfg.ee_frame)
        
        # Add entities to scene
        self.scene.articulations["robot"] = self._robot
        self.scene.rigid_objects["soft_bottle"] = self._soft_bottle
        self.scene.rigid_objects["hard_can"] = self._hard_can
        self.scene.sensors["tactile_sensors"] = self._tactile_sensors
        self.scene.sensors["ee_frame"] = self._ee_frame

        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)

        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # pre-physics step calls

    def _pre_physics_step(self, actions: torch.Tensor):
        """Apply actions to the robot before physics step."""
        self.prev_actions.copy_(self.actions)
        self.actions = actions.clone().clamp(-1.0, 1.0)
        targets = self.robot_dof_targets + self.robot_dof_speed_scales * self.dt * self.actions * self.cfg.action_scale
        self.robot_dof_targets[:] = torch.clamp(targets, self.robot_dof_lower_limits, self.robot_dof_upper_limits)
        
        # Update grasp stability tracking
        self._update_grasp_stability()

    def _apply_action(self):
        """Apply the computed joint targets to the robot."""
        self._robot.set_joint_position_target(self.robot_dof_targets)

    # post-physics step calls

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Check termination and truncation conditions."""
        # Check if objects fell below table
        soft_pos = self._soft_bottle.data.root_pos_w[:, 2]
        hard_pos = self._hard_can.data.root_pos_w[:, 2]
        
        # Terminate if any object falls below table
        terminated = (soft_pos < -0.05) | (hard_pos < -0.05)
        
        # Truncate on timeout
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        
        return terminated, truncated

    def _get_rewards(self) -> torch.Tensor:
        """Compute environment rewards."""
        return compute_rewards(
            self.actions,
            self.prev_actions,
            self._robot.data.joint_pos,
            self._robot.data.joint_vel,
            self._robot.data.root_pos_w,
            self._robot.data.root_quat_w,
            self._ee_frame.data.target_pos_w[..., 0, :],
            self._soft_bottle.data.root_pos_w,
            self._hard_can.data.root_pos_w,
            self._tactile_sensors.data.net_forces_w_history,
            self.object_softness,
            self.target_areas,
            self.stable_grasp_timer,
            self.is_grasping,
            self.grasp_success,
            self.object_lifted,
            self.cfg.tactile_reward_scale,
            self.cfg.force_control_scale,
            self.cfg.sorting_accuracy_scale,
            self.cfg.approach_reward_scale,
            self.cfg.action_penalty_scale,
            self.cfg.joint_vel_penalty_scale,
            self.cfg.drop_penalty_scale,
            self.cfg.stable_grasp_reward_scale,
            self.cfg.minimal_height,
            self.cfg.goal_threshold,
            self.cfg.stable_grasp_time,
            self.cfg.friction_threshold,
        )

    def _reset_idx(self, env_ids: Sequence[int]) -> None:
        """Reset environments at specified indices."""
        if len(env_ids) == 0:
            return

        # Convert to tensor for internal use
        if isinstance(env_ids, list):
            env_ids_tensor = torch.tensor(env_ids, dtype=torch.long, device=self.device)
        else:
            env_ids_tensor = torch.tensor(list(env_ids), dtype=torch.long, device=self.device)

        # robot state
        robot_joint_pos = self._robot.data.default_joint_pos[env_ids_tensor] + sample_uniform(
            -0.125,
            0.125,
            (len(env_ids_tensor), self._robot.num_joints),
            self.device,
        )
        robot_joint_pos = torch.clamp(robot_joint_pos, self.robot_dof_lower_limits, self.robot_dof_upper_limits)
        robot_joint_vel = torch.zeros_like(robot_joint_pos)
        self._robot.set_joint_position_target(robot_joint_pos, env_ids=env_ids_tensor)
        self._robot.write_joint_state_to_sim(robot_joint_pos, robot_joint_vel, None, env_ids_tensor)

        # Update robot DOF targets
        self.robot_dof_targets[env_ids_tensor] = robot_joint_pos

        # Reset objects with random positions
        # Soft bottle
        soft_pos = self._soft_bottle.data.default_root_state[env_ids_tensor, :3].clone()
        soft_pos[:, 0] += sample_uniform(
            self.cfg.object_init_pose_range["x"][0], 
            self.cfg.object_init_pose_range["x"][1], 
            (len(env_ids_tensor),), 
            self.device
        )
        soft_pos[:, 1] += sample_uniform(
            self.cfg.object_init_pose_range["y"][0], 
            self.cfg.object_init_pose_range["y"][1], 
            (len(env_ids_tensor),), 
            self.device
        )
        soft_pos[:, 2] += sample_uniform(
            self.cfg.object_init_pose_range["z"][0], 
            self.cfg.object_init_pose_range["z"][1], 
            (len(env_ids_tensor),), 
            self.device
        )
        soft_pos[:, :3] += self.scene.env_origins[env_ids_tensor]
        
        soft_rot = self._soft_bottle.data.default_root_state[env_ids_tensor, 3:7]
        soft_vel = torch.zeros_like(self._soft_bottle.data.default_root_state[env_ids_tensor, 7:13])
        soft_state = torch.cat([soft_pos, soft_rot, soft_vel], dim=1)
        self._soft_bottle.write_root_state_to_sim(soft_state, env_ids_tensor)

        # Hard can
        hard_pos = self._hard_can.data.default_root_state[env_ids_tensor, :3].clone()
        hard_pos[:, 0] += sample_uniform(
            self.cfg.object_init_pose_range["x"][0], 
            self.cfg.object_init_pose_range["x"][1], 
            (len(env_ids_tensor),), 
            self.device
        )
        hard_pos[:, 1] += sample_uniform(
            self.cfg.object_init_pose_range["y"][0], 
            self.cfg.object_init_pose_range["y"][1], 
            (len(env_ids_tensor),), 
            self.device
        )
        hard_pos[:, 2] += sample_uniform(
            self.cfg.object_init_pose_range["z"][0], 
            self.cfg.object_init_pose_range["z"][1], 
            (len(env_ids_tensor),), 
            self.device
        )
        hard_pos[:, :3] += self.scene.env_origins[env_ids_tensor]
        
        hard_rot = self._hard_can.data.default_root_state[env_ids_tensor, 3:7]
        hard_vel = torch.zeros_like(self._hard_can.data.default_root_state[env_ids_tensor, 7:13])
        hard_state = torch.cat([hard_pos, hard_rot, hard_vel], dim=1)
        self._hard_can.write_root_state_to_sim(hard_state, env_ids_tensor)

        # Sample new object softness labels and target areas
        self.object_softness[env_ids_tensor] = torch.randint(0, 2, (len(env_ids_tensor),), device=self.device)
        self._sample_target_areas(env_ids_tensor)

        # Reset grasp stability tracking
        self.stable_grasp_timer[env_ids_tensor] = 0.0
        self.is_grasping[env_ids_tensor] = False
        self.grasp_success[env_ids_tensor] = False
        self.object_lifted[env_ids_tensor] = False

        # Reset previous actions
        self.prev_actions[env_ids_tensor] = 0.0

    def _get_observations(self) -> dict:
        """Compute environment observations."""
        # joint positions and velocities
        robot_joint_pos_rel = self._robot.data.joint_pos - self._robot.data.default_joint_pos
        robot_joint_vel_rel = self._robot.data.joint_vel * self.cfg.dof_velocity_scale

        # Get end-effector pose
        ee_pos_w = self._ee_frame.data.target_pos_w[..., 0, :]  # end-effector position
        ee_quat_w = self._ee_frame.data.target_quat_w[..., 0, :]  # end-effector orientation
        
        # Convert to robot frame
        ee_pos_rel, ee_quat_rel = subtract_frame_transforms(
            self._robot.data.root_pos_w, self._robot.data.root_quat_w, ee_pos_w, ee_quat_w
        )
        ee_pose = torch.cat([ee_pos_rel, ee_quat_rel], dim=-1)

        # Get object positions relative to robot
        soft_pos_w = self._soft_bottle.data.root_pos_w[:, :3]
        hard_pos_w = self._hard_can.data.root_pos_w[:, :3]
        
        soft_pos_rel, _ = subtract_frame_transforms(
            self._robot.data.root_pos_w, self._robot.data.root_quat_w, soft_pos_w
        )
        hard_pos_rel, _ = subtract_frame_transforms(
            self._robot.data.root_pos_w, self._robot.data.root_quat_w, hard_pos_w
        )

        # Get tactile sensor data (contact forces on gripper fingers)
        contact_forces = self._tactile_sensors.data.net_forces_w_history
        # Process tactile information - get force magnitudes and friction
        force_magnitude = torch.norm(contact_forces[..., :3], dim=-1)  # (num_envs, history_length)
        friction_magnitude = torch.norm(contact_forces[..., :2], dim=-1)  # xy forces (friction)
        tactile_features = torch.cat([
            force_magnitude.mean(dim=-1, keepdim=True),    # average normal force
            force_magnitude.max(dim=-1)[0].unsqueeze(-1),  # max normal force
            friction_magnitude.mean(dim=-1, keepdim=True), # average friction force
            friction_magnitude.max(dim=-1)[0].unsqueeze(-1), # max friction force
            self.stable_grasp_timer.unsqueeze(-1),         # grasp stability time
        ], dim=-1)

        # last action
        actions = self.prev_actions

        obs = torch.cat(
            (
                robot_joint_pos_rel,  # Joint positions (9)
                robot_joint_vel_rel,  # Joint velocities (9) 
                ee_pose,              # End-effector pose (7)
                tactile_features,     # Tactile force features (5)
                soft_pos_rel,         # Soft object position (3)
                hard_pos_rel,         # Hard object position (3)
                self.target_areas,    # Target sorting areas (3)
                actions,              # Previous actions (9)
            ),
            dim=-1,
        )
        return {"policy": obs}

    # auxiliary methods

    def _update_grasp_stability(self) -> None:
        """Update grasp stability tracking based on tactile feedback and object position."""
        # Get tactile sensor data
        contact_forces = self._tactile_sensors.data.net_forces_w_history
        force_magnitude = torch.norm(contact_forces[..., :3], dim=-1)  # (num_envs, history_length)
        current_force = torch.max(force_magnitude, dim=-1)[0]
        
        # Check friction forces (lateral forces indicate good grasp)
        lateral_forces = torch.norm(contact_forces[..., :2], dim=-1)  # xy forces (friction)
        friction_force = torch.max(lateral_forces, dim=-1)[0]
        
        # Get object positions
        soft_pos = self._soft_bottle.data.root_pos_w
        hard_pos = self._hard_can.data.root_pos_w
        ee_pos = self._ee_frame.data.target_pos_w[..., 0, :]
        
        # Determine which object is being grasped
        soft_dist = torch.norm(ee_pos - soft_pos[:, :3], dim=1)
        hard_dist = torch.norm(ee_pos - hard_pos[:, :3], dim=1)
        grasping_soft = soft_dist < hard_dist
        
        # Check if object is lifted above table
        object_height = torch.where(grasping_soft, soft_pos[:, 2], hard_pos[:, 2])
        self.object_lifted = object_height > self.cfg.minimal_height
        
        # Stable grasp criteria:
        # 1. Sufficient normal force (contact)
        # 2. Sufficient friction force (stable grip)
        # 3. Object is lifted
        stable_contact = current_force > 1.0
        stable_friction = friction_force > self.cfg.friction_threshold
        currently_stable = stable_contact & stable_friction & self.object_lifted
        
        # Update grasp status
        self.is_grasping = currently_stable
        
        # Update stable grasp timer
        self.stable_grasp_timer = torch.where(
            currently_stable,
            self.stable_grasp_timer + self.dt,  # increment if stable
            torch.zeros_like(self.stable_grasp_timer)  # reset if not stable
        )
        
        # Mark as successful grasp if held stably for required time
        self.grasp_success = self.stable_grasp_timer >= self.cfg.stable_grasp_time

    def _sample_target_areas(self, env_ids: torch.Tensor | None = None) -> None:
        """Sample target sorting areas for each environment."""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        # Soft area position: [-0.4, 0.4, 0.0] (green area)
        # Hard area position: [-0.4, -0.4, 0.0] (red area)
        soft_area = torch.tensor([-0.4, 0.4, 0.0], device=self.device)
        hard_area = torch.tensor([-0.4, -0.4, 0.0], device=self.device)

        # Assign target based on object softness
        for i, env_id in enumerate(env_ids):
            if self.object_softness[env_id] == 1:  # soft object
                self.target_areas[env_id] = soft_area
            else:  # hard object
                self.target_areas[env_id] = hard_area


# Math utility functions from Isaac Lab

@torch.jit.script
def quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Multiply two quaternions together."""
    if q1.shape != q2.shape:
        msg = f"Expected input quaternion shape mismatch: {q1.shape} != {q2.shape}."
        raise ValueError(msg)
    # reshape to (N, 4) for multiplication
    shape = q1.shape
    q1 = q1.reshape(-1, 4)
    q2 = q2.reshape(-1, 4)
    # extract components from quaternions
    w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
    # perform multiplication
    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
    w = qq - ww + (z1 - y1) * (y2 - z2)
    x = qq - xx + (x1 + w1) * (x2 + w2)
    y = qq - yy + (w1 - x1) * (y2 + z2)
    z = qq - zz + (z1 + y1) * (w2 - x2)

    return torch.stack([w, x, y, z], dim=-1).view(shape)

@torch.jit.script
def quat_apply(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """Apply a quaternion rotation to a vector."""
    # store shape
    shape = vec.shape
    # reshape to (N, 3) for multiplication
    quat = quat.reshape(-1, 4)
    vec = vec.reshape(-1, 3)
    # extract components from quaternions
    xyz = quat[:, 1:]
    t = xyz.cross(vec, dim=-1) * 2
    return (vec + quat[:, 0:1] * t + xyz.cross(t, dim=-1)).view(shape)

@torch.jit.script
def quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    """Computes the conjugate of a quaternion."""
    shape = q.shape
    q = q.reshape(-1, 4)
    return torch.cat((q[..., 0:1], -q[..., 1:]), dim=-1).view(shape)

@torch.jit.script
def quat_inv(q: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """Computes the inverse of a quaternion."""
    return quat_conjugate(q) / q.pow(2).sum(dim=-1, keepdim=True).clamp(min=eps)

@torch.jit.script
def subtract_frame_transforms(
    t01: torch.Tensor, q01: torch.Tensor, t02: torch.Tensor | None = None, q02: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Subtract transformations between two reference frames into a stationary frame."""
    # compute orientation
    q10 = quat_inv(q01)
    if q02 is not None:
        q12 = quat_mul(q10, q02)
    else:
        q12 = q10
    # compute translation
    if t02 is not None:
        t12 = quat_apply(q10, t02 - t01)
    else:
        t12 = quat_apply(q10, -t01)
    return t12, q12

@torch.jit.script
def combine_frame_transforms(
    t01: torch.Tensor, q01: torch.Tensor, t12: torch.Tensor | None = None, q12: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Combine transformations between two reference frames into a stationary frame."""
    # compute orientation
    if q12 is not None:
        q02 = quat_mul(q01, q12)
    else:
        q02 = q01
    # compute translation
    if t12 is not None:
        t02 = t01 + quat_apply(q01, t12)
    else:
        t02 = t01

    return t02, q02

@torch.jit.script
def compute_rewards(
    actions: torch.Tensor,
    prev_actions: torch.Tensor,
    robot_joint_pos: torch.Tensor,
    robot_joint_vel: torch.Tensor,
    robot_root_pos: torch.Tensor,
    robot_root_quat: torch.Tensor,
    ee_pos: torch.Tensor,
    soft_object_pos: torch.Tensor,
    hard_object_pos: torch.Tensor,
    tactile_forces: torch.Tensor,
    object_softness: torch.Tensor,
    target_areas: torch.Tensor,
    stable_grasp_timer: torch.Tensor,
    is_grasping: torch.Tensor,
    grasp_success: torch.Tensor,
    object_lifted: torch.Tensor,
    tactile_reward_scale: float,
    force_control_scale: float,
    sorting_accuracy_scale: float,
    approach_reward_scale: float,
    action_penalty_scale: float,
    joint_vel_penalty_scale: float,
    drop_penalty_scale: float,
    stable_grasp_reward_scale: float,
    minimal_height: float,
    goal_threshold: float,
    stable_grasp_time: float,
    friction_threshold: float,
) -> torch.Tensor:
    """Compute tactile-based rewards for object sorting task with drop penalty and stable grasping."""
    
    # Process tactile sensor data
    force_magnitude = torch.norm(tactile_forces[..., :3], dim=-1)  # (num_envs, history_length)
    friction_magnitude = torch.norm(tactile_forces[..., :2], dim=-1)  # xy forces (friction)
    current_force = torch.max(force_magnitude, dim=-1)[0]
    current_friction = torch.max(friction_magnitude, dim=-1)[0]
    
    # Tactile grasping reward - reward for stable contact with sufficient friction
    stable_contact = current_force > 1.0
    stable_friction = current_friction > friction_threshold
    tactile_reward = torch.where(stable_contact & stable_friction, 1.0, 0.0)

    # Determine which object is being grasped
    soft_dist = torch.norm(ee_pos - soft_object_pos[:, :3], dim=1)
    hard_dist = torch.norm(ee_pos - hard_object_pos[:, :3], dim=1)
    grasping_soft = soft_dist < hard_dist
    
    # Force control reward - encourage appropriate force based on object type
    force_control_reward = torch.zeros(len(actions), device=actions.device)
    # Soft objects: reward if force <= 5.0N
    soft_mask = grasping_soft & (current_force > 0.1)
    force_control_reward[soft_mask] = torch.where(
        current_force[soft_mask] <= 5.0, 1.0, -0.1 * (current_force[soft_mask] - 5.0)
    )
    # Hard objects: reward if force <= 20.0N
    hard_mask = (~grasping_soft) & (current_force > 0.1)
    force_control_reward[hard_mask] = torch.where(
        current_force[hard_mask] <= 20.0, 1.0, -0.1 * (current_force[hard_mask] - 20.0)
    )

    # Drop penalty - heavily penalize if objects fall below table
    drop_penalty = torch.zeros(len(actions), device=actions.device)
    soft_dropped = soft_object_pos[:, 2] < minimal_height
    hard_dropped = hard_object_pos[:, 2] < minimal_height
    drop_penalty = torch.where(soft_dropped | hard_dropped, 1.0, 0.0)

    # Stable grasp reward - progressive reward for holding object stably
    stable_grasp_reward = torch.zeros(len(actions), device=actions.device)
    # Progressive reward based on how long object has been held stably
    stable_grasp_reward = torch.where(
        is_grasping & object_lifted,
        torch.clamp(stable_grasp_timer / stable_grasp_time, 0.0, 1.0),  # 0 to 1 based on time
        torch.zeros_like(stable_grasp_timer)
    )
    # Bonus reward for successful grasp (held for full duration)
    stable_grasp_reward = torch.where(grasp_success, stable_grasp_reward + 2.0, stable_grasp_reward)

    # Sorting accuracy reward - only give if object is stably grasped
    soft_area_pos = torch.tensor([-0.4, 0.4, 0.0], device=actions.device).expand(len(actions), -1)
    hard_area_pos = torch.tensor([-0.4, -0.4, 0.0], device=actions.device).expand(len(actions), -1)
    
    soft_to_soft_area = torch.norm(soft_object_pos[:, :2] - soft_area_pos[:, :2], dim=1)
    hard_to_hard_area = torch.norm(hard_object_pos[:, :2] - hard_area_pos[:, :2], dim=1)
    
    sorting_reward = torch.zeros(len(actions), device=actions.device)
    # Only reward sorting if object is successfully grasped
    sorting_reward = torch.where(
        grasp_success,
        ((soft_to_soft_area < goal_threshold).float() + (hard_to_hard_area < goal_threshold).float()) * 3.0,
        torch.zeros_like(sorting_reward)
    )

    # Approach reward - encourage moving towards objects (only if not already grasping)
    min_dist = torch.min(soft_dist, hard_dist)
    approach_reward = torch.where(
        ~is_grasping,  # only reward approach if not currently grasping
        torch.exp(-min_dist * 2.0),
        torch.zeros_like(min_dist)
    )

    # Penalties
    action_penalty = torch.sum(torch.square(actions - prev_actions), dim=1)
    joint_vel_penalty = torch.sum(torch.square(robot_joint_vel), dim=1)

    # total reward
    total_reward = (
        tactile_reward_scale * tactile_reward
        + force_control_scale * force_control_reward
        + sorting_accuracy_scale * sorting_reward
        + approach_reward_scale * approach_reward
        + stable_grasp_reward_scale * stable_grasp_reward
        + action_penalty_scale * action_penalty
        + joint_vel_penalty_scale * joint_vel_penalty
        + drop_penalty_scale * drop_penalty  # negative penalty for dropping
    )

    return total_reward
