# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Test Locomotion Tracking (Nav1 Guided | Single Env | Live Plotting)."""

import argparse
import os
import types
import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import deque

from isaaclab.app import AppLauncher
import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Live visualization of Nav and Loco commands.")
parser.add_argument("--disable_fabric", action="store_true", default=False)
# 强制默认环境数为 1
parser.add_argument("--num_envs", type=int, default=1, help="Must be 1 for live plotting.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import go2_lidar.tasks  # noqa: F401
import gymnasium as gym
import isaaclab.utils.math as math_utils
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
from rsl_rl.runners import OnPolicyRunner

def main():
    task_name = "Unitree-Go2-Navigation"
    print("=" * 50)
    print("Starting Live Plotting Test (Single Env | Global Random Spawn & 10m Goal)...")
    print("=" * 50)

    # 确保只使用 1 个环境
    args_cli.num_envs = 1

    env_cfg = parse_env_cfg(
        task_name, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    agent_cfg = cli_args.parse_rsl_rl_cfg(task_name, args_cli)

    # 环境配置覆盖
    env_cfg.is_play_env = True  # 打开 Play 模式以便在仿真器中观察
    env_cfg.events.push_robot = None
    env_cfg.no_obstacle = True              
    env_cfg.use_dynamic_obstacle = False
    env_cfg.static_friction_range = (0.4, 1.0)
    env_cfg.dynamic_friction_range = (0.4, 1.0)
    env_cfg.episode_length_s = 15.0  # 放宽单次 episode 时长，方便观察连续过程
    env_cfg.set_raycaster_measure_pattern("11x")

    agent_cfg.load_run = "nav_1"
    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    
    env = gym.make(task_name, cfg=env_cfg, render_mode=None)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    base_env = env.unwrapped

    # =========================================================================
    # 重写底层重置函数 (保留全局随机与 10m 目标逻辑)
    # =========================================================================
    def custom_reset_robot_and_cmd(self, env_ids):
        if len(env_ids) == 0:
            return

        root_state = self._robot.data.default_root_state[env_ids].clone()
        rand_x = torch.rand(len(env_ids), device=self.device) * 80.0 - 40.0
        rand_y = torch.rand(len(env_ids), device=self.device) * 80.0 - 40.0
        root_state[:, 0] = rand_x
        root_state[:, 1] = rand_y
        
        zeros = torch.zeros(len(env_ids), device=self.device)
        rand_yaw = torch.rand(len(env_ids), device=self.device) * 2 * np.pi - np.pi
        root_state[:, 3:7] = math_utils.quat_from_euler_xyz(zeros, zeros, rand_yaw)
        
        self._robot.write_root_link_pose_to_sim(root_state[:, :7], env_ids)
        self._robot.write_root_com_velocity_to_sim(root_state[:, 7:], env_ids)
        
        self._robot_start_pos[env_ids, :2] = root_state[:, :2]
        self._robot_start_pos[env_ids, 2] = 0.0

        angles = torch.rand(len(env_ids), device=self.device) * 2 * np.pi
        distances = torch.rand(len(env_ids), device=self.device) * 8.0 + 2.0 
        rand_gx = torch.clamp(rand_x + distances * torch.cos(angles), -39.0, 39.0)
        rand_gy = torch.clamp(rand_y + distances * torch.sin(angles), -39.0, 39.0)

        self._cmd_goal_w[env_ids, 0] = rand_gx
        self._cmd_goal_w[env_ids, 1] = rand_gy
        self._cmd_goal_w[env_ids, 2] = 0.0

        self._cmd_goal_b[env_ids] = math_utils.quat_apply_inverse(
            math_utils.yaw_quat(self._robot.data.root_quat_w[env_ids]),
            self._cmd_goal_w[env_ids] - self._robot.data.root_pos_w[env_ids]
        )
        cmd_goal_dir_norm = torch.norm(self._cmd_goal_b[env_ids], dim=-1)
        self._cmd_goal_dir_b[env_ids] = self._cmd_goal_b[env_ids] / (cmd_goal_dir_norm.unsqueeze(-1) + 1e-6)
        mask = cmd_goal_dir_norm < 0.1
        self._cmd_goal_dir_b[env_ids[mask]] = 0.0
        self._cmd_goal_heading_b[env_ids] = torch.atan2(
            self._cmd_goal_dir_b[env_ids, 1], self._cmd_goal_dir_b[env_ids, 0]
        ).unsqueeze(-1)

    base_env._reset_robot_and_cmd = types.MethodType(custom_reset_robot_and_cmd, base_env)
    base_env._env_type[:] = 2
    base_env._reset_idx(torch.arange(args_cli.num_envs, device=args_cli.device))
    obs, _ = env.get_observations()

    # 加载模型
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), device=agent_cfg.device, use_ray=True)
    ppo_runner.load(resume_path, load_optimizer=False)
    nav_policy = ppo_runner.get_inference_policy(device=base_env.device)

    # =========================================================================
    # 实时绘图 (Live Plotting) 初始化
    # =========================================================================
    plt.ion()  # 开启交互模式
    fig, axs = plt.subplots(3, 1, figsize=(10, 8))
    fig.canvas.manager.set_window_title('Real-time Velocity Tracking')
    
    plot_history_len = 150 # 界面显示最近 150 步的数据 (约 3 秒)
    time_data = deque(maxlen=plot_history_len)
    
    nav_qs = [deque(maxlen=plot_history_len) for _ in range(3)]
    cmd_qs = [deque(maxlen=plot_history_len) for _ in range(3)]
    true_qs = [deque(maxlen=plot_history_len) for _ in range(3)]
    
    lines_nav, lines_cmd, lines_true = [], [], []
    titles = ['X Velocity (m/s)', 'Y Velocity (m/s)', 'Yaw Velocity (rad/s)']
    
    for i in range(3):
        axs[i].set_title(titles[i], fontweight='bold')
        axs[i].grid(True, linestyle='--', alpha=0.6)
        
        ln_nav, = axs[i].plot([], [], label='Nav Action (Raw)', color='gray', linestyle=':')
        ln_cmd, = axs[i].plot([], [], label='Scaled Cmd (To Loco)', color='blue', linestyle='--')
        ln_true, = axs[i].plot([], [], label='True Velocity', color='red', linestyle='-')
        
        lines_nav.append(ln_nav)
        lines_cmd.append(ln_cmd)
        lines_true.append(ln_true)
        axs[i].legend(loc='upper right')

    plt.tight_layout()

    # =========================================================================
    # 仿真主循环
    # =========================================================================
    step_counter = 0
    loco_cmd_scale = torch.tensor([[2.0, 2.0, 0.25]], device=args_cli.device)

    while simulation_app.is_running():
        with torch.inference_mode():
            nav_actions = nav_policy(obs)
            
            # 提取 3 组数据 (转为 CPU numpy 便于画图)
            nav_np = nav_actions[0].cpu().numpy()
            
            cmd_vel_scaled = nav_actions.clone() * loco_cmd_scale
            cmd_np = cmd_vel_scaled[0].cpu().numpy()
            
            robot_data = base_env._robot.data
            true_vel_xyw = torch.cat([robot_data.root_com_lin_vel_b[:, :2], robot_data.root_com_ang_vel_b[:, 2:3]], dim=-1)
            true_np = true_vel_xyw[0].cpu().numpy()

            # 数据入队
            time_data.append(step_counter)
            for i in range(3):
                nav_qs[i].append(nav_np[i])
                cmd_qs[i].append(cmd_np[i])
                true_qs[i].append(true_np[i])

            # 每 5 步刷新一次图表 (防止 matplotlib 拖慢仿真速度)
            if step_counter % 5 == 0:
                for i in range(3):
                    lines_nav[i].set_data(time_data, nav_qs[i])
                    lines_cmd[i].set_data(time_data, cmd_qs[i])
                    lines_true[i].set_data(time_data, true_qs[i])
                    
                    # 动态调整坐标轴
                    axs[i].set_xlim(min(time_data), max(time_data) + 1)
                    all_vals = list(nav_qs[i]) + list(cmd_qs[i]) + list(true_qs[i])
                    if all_vals:
                        y_min, y_max = min(all_vals), max(all_vals)
                        axs[i].set_ylim(y_min - 0.2, y_max + 0.2)
                        
                fig.canvas.draw()
                fig.canvas.flush_events()

            # 步进环境
            obs, _, dones, _ = env.step(nav_actions)
            step_counter += 1

            # 目标到达检测与重置逻辑
            robot_pos_2d = base_env._robot.data.root_pos_w[:, :2]
            goal_pos_2d = base_env._cmd_goal_w[:, :2]
            dist_to_goal = torch.norm(goal_pos_2d - robot_pos_2d, dim=-1)
            
            reached_mask = dist_to_goal < 0.3 
            reset_mask = dones | reached_mask

            if reset_mask.any():
                reset_env_ids = reset_mask.nonzero(as_tuple=False).squeeze(-1)
                manual_reset_ids = (reached_mask & ~dones).nonzero(as_tuple=False).squeeze(-1)
                
                if len(manual_reset_ids) > 0:
                    base_env._reset_idx(manual_reset_ids)

                if hasattr(base_env, "_reset_physx_materials"):
                    base_env._reset_physx_materials(reset_env_ids.cpu())

if __name__ == "__main__":
    main()