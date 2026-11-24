# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Test Navigation Policy Tracking and Plot Results (Fixed Scenario)."""

import argparse
import os
import time
import numpy as np
import matplotlib.pyplot as plt

from isaaclab.app import AppLauncher
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Test trained nav policy with a fixed scenario.")
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--use_pred_rays", action="store_true", default=False)
parser.add_argument("--num_ray_centers", type=str, default="11x", choices=["1x", "3x", "5x", "11x"])

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import go2_lidar.tasks  # noqa: F401
import gymnasium as gym
import torch
import isaaclab.utils.math as math_utils 
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
from rsl_rl.runners import OnPolicyRunner
from gymnasium.wrappers import RecordVideo

import omni.usd
from pxr import UsdPhysics

def main():
    task_name = "Unitree-Go2-Navigation"
    print("=" * 50)
    print("Testing Navigation Policy [nav_res] with Online Gradient MPC.")
    print("=" * 50)

    # 1. 解析基础配置
    env_cfg = parse_env_cfg(
        task_name,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    agent_cfg = cli_args.parse_rsl_rl_cfg(task_name, args_cli)

    # -----------------------------------------------------------------
    # 强制纯平地与无障碍测试环境
    # -----------------------------------------------------------------
    env_cfg.is_play_env = True
    env_cfg.events = None 
    env_cfg.no_obstacle = True               
    env_cfg.use_dynamic_obstacle = False     

    if hasattr(env_cfg, "terrain"):
        env_cfg.terrain.terrain_type = "plane"
        class DummyTerrainGenerator:
            num_rows = 10    
            num_cols = 30    
            size = (10.0, 10.0)
        env_cfg.terrain.terrain_generator = DummyTerrainGenerator()
        
    from isaaclab.terrains import TerrainImporter
    original_init = TerrainImporter.__init__
    
    def patched_terrain_init(self, cfg, **kwargs):
        original_init(self, cfg, **kwargs)
        device = getattr(self, "device", "cuda:0")
        if getattr(self, "terrain_origins", None) is None:
            self.terrain_origins = torch.zeros((10, 30, 3), device=device)
        if getattr(self, "terrain_levels", None) is None:
            self.terrain_levels = torch.zeros((args_cli.num_envs,), dtype=torch.long, device=device)
        if getattr(self, "env_origins", None) is None:
            self.env_origins = torch.zeros((args_cli.num_envs, 3), device=device)
            
    TerrainImporter.__init__ = patched_terrain_init
    # -----------------------------------------------------------------

    agent_cfg.load_run = "nav_1"
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    print(f"[INFO] Loaded Nav Policy from: {resume_path}")

    env_cfg.use_predicted_rays = args_cli.use_pred_rays
    env_cfg.set_raycaster_measure_pattern(args_cli.num_ray_centers)

    env = gym.make(task_name, cfg=env_cfg, render_mode=None)
    
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # =========================================================================
    # 【新增】：强制固定出生点和朝向，防止起步绕圈！
    # =========================================================================    
    base_env = env.unwrapped
    fixed_goal_x = env.unwrapped._robot.data.root_pos_w[:, 0] + 20
    fixed_goal_y = env.unwrapped._robot.data.root_pos_w[:, 1]
    import types
    def custom_reset_robot_and_cmd(self, env_ids):
        if len(env_ids) == 0:
            return

        root_state = self._robot.data.default_root_state[env_ids].clone()
        
        # 1. 固定出生在坐标原点 (X=0, Y=0)
        root_state[:, 0] = 0.0
        root_state[:, 1] = 0.0
        
        # 2. 固定面朝正前方 (Yaw=0，即 X 轴正方向)
        zeros = torch.zeros(len(env_ids), device=self.device)
        root_state[:, 3:7] = math_utils.quat_from_euler_xyz(zeros, zeros, zeros) 
        
        self._robot.write_root_link_pose_to_sim(root_state[:, :7], env_ids)
        self._robot.write_root_com_velocity_to_sim(root_state[:, 7:], env_ids)
        
        self._robot_start_pos[env_ids, :2] = root_state[:, :2]
        self._robot_start_pos[env_ids, 2] = 0.0

        # 3. 初始化目标点
        self._cmd_goal_w[env_ids, 0] = fixed_goal_x
        self._cmd_goal_w[env_ids, 1] = fixed_goal_y
        self._cmd_goal_w[env_ids, 2] = 0.0

        # 刷新局部导航坐标
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
    
    # 强制执行一次重置，让我们的固定原点和朝向立刻生效
    base_env._reset_idx(torch.arange(args_cli.num_envs, device=args_cli.device))
    obs, _ = env.get_observations()

    ppo_runner = OnPolicyRunner(
        env,
        agent_cfg.to_dict(),
        log_dir=None,
        device=agent_cfg.device,
        use_ray=True,
    )
    ppo_runner.load(resume_path, load_optimizer=False)
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    max_steps = 600      
    mid_step = 300       
    
    time_axis = []
    v_cmd_x_list, v_true_x_list = [], []
    v_cmd_y_list, v_true_y_list = [], []
    v_optimized_x_list, v_optimized_y_list = [], []  # 经过梯度下降的真实V_cmd
    
    # 存储预测误差
    err_x_list, err_y_list, err_norm_list = [], [], []
    
    obs, _ = env.get_observations()
    dt = env.unwrapped.step_dt

    print("Starting simulation loop...")
    for step in range(max_steps):
        with torch.inference_mode():
            # [全局直线引导]: 目标点始终位于全局正X前方5米，防止绕圈
            robot_pos = env.unwrapped._robot.data.root_pos_w
            env.unwrapped._cmd_goal_w[:, 0] = fixed_goal_x#robot_pos[:, 0] + 5.0  
            env.unwrapped._cmd_goal_w[:, 1] = fixed_goal_y                   

            actions = policy(obs)
            
            # [突变逻辑]：在仿真中期瞬间使得地面打滑
            if step == mid_step:
                stage = omni.usd.get_context().get_stage()
                for prim in stage.Traverse():
                    if prim.HasAPI(UsdPhysics.MaterialAPI):
                        material = UsdPhysics.MaterialAPI(prim)
                        material.GetStaticFrictionAttr().Set(0.45)
                        material.GetDynamicFrictionAttr().Set(0.45)
                        # material.GetRestitutionAttr().Set(0.1) 
                print("[!] Environment parameters updated. Watch for SNS intervention!\n")

            # 提取 extras 中的预测误差数据
            obs, _, _, extras = env.step(actions)

            cmd_vx = actions[0, 0].item() 
            cmd_vy = actions[0, 1].item()
            true_vx = env.unwrapped._robot.data.root_com_lin_vel_b[0, 0].item()
            true_vy = env.unwrapped._robot.data.root_com_lin_vel_b[0, 1].item()
            
            # 提取经过梯度下降的真实V_cmd
            real_high_actions = extras["real_high_actions"][0].cpu().numpy()
            opt_vx = real_high_actions[0]
            opt_vy = real_high_actions[1]
            
            # 记录误差
            pred_err = extras["predicted_vel_error"][0].cpu().numpy()
            pred_norm = extras["predicted_error_norm"][0].item()

            time_axis.append(step * dt)
            v_cmd_x_list.append(cmd_vx)
            v_true_x_list.append(true_vx)
            v_optimized_x_list.append(opt_vx)
            v_cmd_y_list.append(cmd_vy)
            v_true_y_list.append(true_vy)
            v_optimized_y_list.append(opt_vy)
            
            err_x_list.append(pred_err[0])
            err_y_list.append(pred_err[1])
            err_norm_list.append(pred_norm)
    print("Simulation finished. Saving raw plot data...")
    data_save_path = "sns_online_mpc_plot_data.npz"
    # 确保 goal_x 和 goal_y 是普通的 Python float，而不是 GPU Tensor
    save_goal_x = fixed_goal_x.item() if torch.is_tensor(fixed_goal_x) else fixed_goal_x
    save_goal_y = fixed_goal_y.item() if torch.is_tensor(fixed_goal_y) else fixed_goal_y
    np.savez(
        data_save_path,
        time_axis=time_axis,
        v_cmd_x=v_cmd_x_list,     v_cmd_y=v_cmd_y_list,
        v_opt_x=v_optimized_x_list, v_opt_y=v_optimized_y_list,
        v_true_x=v_true_x_list,   v_true_y=v_true_y_list,
        err_x=err_x_list,         err_y=err_y_list,
        err_norm=err_norm_list,
        mid_step=mid_step,
        fixed_goal_x=save_goal_x, fixed_goal_y=save_goal_y
    )
    print(f"Raw plot data successfully saved to: {data_save_path}")
    # 5. 绘图导出 (3个子图)
    print("Simulation finished. Plotting results...")
    plt.figure(figsize=(12, 12))

    # 子图1：X速度
    plt.subplot(3, 1, 1)
    plt.plot(time_axis, v_cmd_x_list, label="Nav Command ($v_{cmd\_x}$)", color="blue", linestyle="--", linewidth=2)
    plt.plot(time_axis, v_optimized_x_list, label="Optimized ($v_{opt\_x}$)", color="purple", linestyle="-.", linewidth=2)
    plt.plot(time_axis, v_true_x_list, label="True Velocity ($v_{true\_x}$)", color="orange", linewidth=2)
    plt.axvline(x=time_axis[mid_step], color='red', linestyle=':', linewidth=2, label="Friction Drop")
    plt.title("X-Axis Velocity Tracking")
    plt.ylabel("Velocity (m/s)")
    plt.legend()
    plt.grid(True)

    # 子图2：Y速度
    plt.subplot(3, 1, 2)
    plt.plot(time_axis, v_cmd_y_list, label="Nav Command ($v_{cmd\_y}$)", color="blue", linestyle="--", linewidth=2)
    plt.plot(time_axis, v_optimized_y_list, label="Optimized ($v_{opt\_y}$)", color="purple", linestyle="-.", linewidth=2)
    plt.plot(time_axis, v_true_y_list, label="True Velocity ($v_{true\_y}$)", color="orange", linewidth=2)
    plt.axvline(x=time_axis[mid_step], color='red', linestyle=':', linewidth=2, label="Friction Drop")
    plt.title("Y-Axis Velocity Tracking")
    plt.ylabel("Velocity (m/s)")
    plt.legend()
    plt.grid(True)

    # 子图3：SNS预测误差
    plt.subplot(3, 1, 3)
    plt.plot(time_axis, err_norm_list, label="Predicted Error Norm ($||e||$)", color="red", linewidth=2)
    plt.plot(time_axis, err_x_list, label="Predicted $v_x$ Error", color="blue", alpha=0.5)
    plt.plot(time_axis, err_y_list, label="Predicted $v_y$ Error", color="green", alpha=0.5)
    plt.axhline(y=0.5, color='black', linestyle='--', linewidth=2, label="Intervention Threshold (0.5)")
    plt.axvline(x=time_axis[mid_step], color='red', linestyle=':', linewidth=2, label="Friction Drop")
    
    # 标出触发梯度的区间
    plt.fill_between(time_axis, 0, err_norm_list, where=(np.array(err_norm_list) > 0.5), 
                     color='red', alpha=0.2, label="Active Gradient Updates")
                     
    plt.title("SNS Predicted Tracking Error (Online MPC Trigger)")
    plt.xlabel("Time (s)")
    plt.ylabel("Error Magnitude")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plot_path = "sns_online_mpc_response.png"
    plt.savefig(plot_path, dpi=300)
    print(f"Plot saved successfully to: {plot_path}")

    env.close()
    simulation_app.close()

if __name__ == "__main__":
    main()