# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch Isaac Sim Simulator first."""

import argparse
import traceback

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False)
parser.add_argument("--video_length", type=int, default=200)
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--num_envs", type=int, default=3)
# --- 新增参数 ---
parser.add_argument("--test_steps", type=int, default=1000, help="Number of steps to run the simulation for variance testing.")
parser.add_argument("--mc_samples", type=int, default=10, help="Number of MC Dropout samples per step.")

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# set task name
task_name = "Unitree-Go2-Locomotion"
print("=" * 50)
print("Playing locomotion policy with MC Dropout Variance Trace.")
print("=" * 50)

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Start playing after the app is launched."""

import os
import time

import go2_lidar.tasks  # noqa: F401
import gymnasium as gym
import torch
import numpy as np
import matplotlib.pyplot as plt

from isaaclab.utils.dict import print_dict
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
from rsl_rl.runners import OnPolicyRunnerLoco


def main():
    # parse configuration
    env_cfg = parse_env_cfg(
        task_name,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    agent_cfg = cli_args.parse_rsl_rl_cfg(task_name, args_cli)

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")

    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    log_dir = os.path.dirname(resume_path)
    print(f"resume path: {resume_path}")
    print(f"log dir: {log_dir}")

    # update env cfg
    env_cfg.is_play_env = True

    # create the environment
    env = gym.make(task_name, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # load previously trained model
    ppo_runner = OnPolicyRunnerLoco(
        env,
        agent_cfg.to_dict(),
        log_dir=None,
        device=agent_cfg.device,
    )
    ppo_runner.load(resume_path, load_optimizer=False)

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # =========================================================================
    # 【最关键的一步】：强行开启底层 ActorCritic 网络 的 train() 模式！
    # 这样才能激活网络内部的 nn.Dropout 层，否则它会自动关闭导致方差为 0。
    # =========================================================================
    ppo_runner.alg.policy.train()
    print(f"[INFO] MC Dropout activated! Sampling {args_cli.mc_samples} times per step.")

    obs, _ = env.get_observations()
    timestep = 0
    trace_vars_history = []

# simulate environment
    while simulation_app.is_running() and timestep < args_cli.test_steps:
        # run everything in inference mode
        with torch.inference_mode():
            
            # --- 1. MC Dropout 推理部分 (保持你现在的代码不变) ---
            norm_obs = ppo_runner.obs_normalizer(obs)
            input_a = ppo_runner.alg.policy.memory_a(norm_obs).squeeze(0)
            
            actions_list = []
            for _ in range(args_cli.mc_samples):
                actions_mean = ppo_runner.alg.policy.actor(input_a)
                actions_list.append(actions_mean)
            
            stacked_actions = torch.stack(actions_list, dim=0)
            action_var = torch.var(stacked_actions, dim=0)
            trace_var = torch.sum(action_var, dim=-1)
            trace_vars_history.append(trace_var.cpu().numpy())

            env_actions = torch.mean(stacked_actions, dim=0)
            obs, _, _, _ = env.step(env_actions)

        # =================================================================
        # 【新增】：在第 500 步瞬间发动 OOD 攻击，改变物理材质
        # =================================================================
        if timestep == 500:
            print("\n[WARNING] Step 500 reached! Injecting OOD Terrain: Switching to FOAM material...\n")
            
            # 穿透 wrapper，直接拿到基础环境和机器人对象
            base_env = env.unwrapped
            robot = base_env._robot
            
            # 获取当前的物理材质
            materials = robot.root_physx_view.get_material_properties()
            
            # 泡沫材质：极低摩擦 (0.05) 模拟打滑
            foam_mat = torch.tensor([0.2, 0.2, 0.0], dtype=materials.dtype, device=materials.device)
            
            # 给所有环境里的狗同时换上泡沫鞋底
            for i in range(base_env.num_envs):
                materials[i, :] = foam_mat.reshape(-1, 1, 3)
                
            # Isaac Sim 要求传 CPU tensor 给 env_ids
            env_ids = torch.arange(base_env.num_envs, device="cpu")
            robot.root_physx_view.set_material_properties(materials, env_ids)
        # =================================================================

        timestep += 1
    # =========================================================================
    # 绘图和保存数据 (必须在关闭 env 之前执行)
    # =========================================================================
    if len(trace_vars_history) > 0:
        print("\n[INFO] Saving and plotting variance trace matrix...")
        # 形状: (test_steps, num_envs)
        trace_matrix = np.stack(trace_vars_history, axis=0) 
        
        # 保存原始矩阵数据
        npy_path = os.path.join(log_dir, "mc_dropout_variance_trace.npy")
        np.save(npy_path, trace_matrix)
        print(f"[INFO] Saved variance matrix to: {npy_path}")

        # 绘制并保存图片
        plt.figure(figsize=(12, 6))
        # 我们将矩阵转置，使得 X 轴为时间(Step)，Y 轴为环境 ID (Env)
        plt.imshow(trace_matrix.T, aspect='auto', cmap='viridis', origin='lower')
        plt.colorbar(label='Trace of Action Variance')
        plt.xlabel('Time Step')
        plt.ylabel('Environment ID')
        plt.title(f'Action Variance Trace using MC Dropout ({args_cli.mc_samples} samples)')
        
        png_path = os.path.join(log_dir, "mc_dropout_variance_trace.png")
        plt.savefig(png_path, dpi=300, bbox_inches='tight')
        print(f"[INFO] Saved variance plot to: {png_path}")

    # close the simulator
    print("[INFO] Closing environment and simulator...")
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()