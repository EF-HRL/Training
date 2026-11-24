# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run a fixed body-frame forward locomotion command and report world-Y drift."""

import argparse
import os

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip


parser = argparse.ArgumentParser(description="Test locomotion with a fixed body-frame forward velocity command.")
parser.add_argument("--video", action="store_true", default=False, help="Record a video of the run.")
parser.add_argument("--video_length", type=int, default=400, help="Number of simulation steps to record.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable Fabric and use USD I/O.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to visualize.")
parser.add_argument("--test_steps", type=int, default=2000, help="Number of simulation steps to run.")
parser.add_argument("--forward_speed", type=float, default=1.0, help="Fixed forward speed in m/s.")
parser.add_argument("--print_interval", type=int, default=100, help="Print measured base velocity every N steps.")

cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.video:
    args_cli.enable_cameras = True

task_name = "Unitree-Go2-Locomotion"
print("=" * 50)
print(f"Testing locomotion with fixed forward command: {args_cli.forward_speed:.2f} m/s")
print("=" * 50)

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import go2_lidar.tasks  # noqa: E402, F401
import gymnasium as gym  # noqa: E402
import isaaclab.terrains as terrain_gen  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from isaaclab.utils.dict import print_dict  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg  # noqa: E402
from rsl_rl.utils import resolve_nn_activation  # noqa: E402
from rsl_rl.runners import OnPolicyRunnerLoco  # noqa: E402


def set_body_forward_command(env: RslRlVecEnvWrapper, forward_speed: float):
    """Force all locomotion envs to use a fixed forward command in the robot body frame."""
    base_env = env.unwrapped
    base_env._cmd_lin_vel[:, 0] = forward_speed
    base_env._cmd_lin_vel[:, 1] = 0.0
    base_env._cmd_ang_vel[:, 0] = 0.0
    base_env._cmd_resample_accums.zero_()


def make_actor_mlp(input_dim: int, hidden_dims: list[int], output_dim: int, activation_name: str) -> nn.Sequential:
    """Build the original RSL-RL actor MLP layout without Dropout layers."""
    activation = resolve_nn_activation(activation_name)
    layers = [nn.Linear(input_dim, hidden_dims[0]), activation]
    for layer_index in range(len(hidden_dims)):
        if layer_index == len(hidden_dims) - 1:
            layers.append(nn.Linear(hidden_dims[layer_index], output_dim))
        else:
            layers.append(nn.Linear(hidden_dims[layer_index], hidden_dims[layer_index + 1]))
            layers.append(activation)
    return nn.Sequential(*layers)


def adapt_policy_actor_to_checkpoint(ppo_runner: OnPolicyRunnerLoco, checkpoint_path: str, agent_cfg):
    """Match the actor module layout to legacy checkpoints saved before Dropout was added."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_state = checkpoint["model_state_dict"]

    has_legacy_actor = "actor.2.weight" in model_state and "actor.4.weight" in model_state
    has_dropout_actor = "actor.3.weight" in model_state and "actor.9.weight" in model_state
    current_actor_state = ppo_runner.alg.policy.actor.state_dict()

    if has_legacy_actor and "3.weight" in current_actor_state:
        rnn_hidden_dim = getattr(agent_cfg.policy, "rnn_hidden_dim", 256)
        ppo_runner.alg.policy.actor = make_actor_mlp(
            input_dim=rnn_hidden_dim,
            hidden_dims=list(agent_cfg.policy.actor_hidden_dims),
            output_dim=ppo_runner.env.num_actions,
            activation_name=agent_cfg.policy.activation,
        ).to(ppo_runner.device)
        print("[INFO] Detected legacy checkpoint actor layout. Using non-Dropout actor for this test run.")
    elif has_dropout_actor:
        print("[INFO] Detected Dropout checkpoint actor layout. Keeping current actor layout.")


def reset_robot_for_trial(env: RslRlVecEnvWrapper, ppo_runner: OnPolicyRunnerLoco) -> torch.Tensor:
    """Reset the env, then put every robot at its terrain origin with zero yaw."""
    obs, _ = env.reset()
    base_env = env.unwrapped
    env_ids = torch.arange(base_env.num_envs, device=base_env.device)

    ppo_runner.alg.policy.reset()
    base_env.episode_length_buf[:] = 0
    base_env._actions.zero_()
    for actions in base_env._previous_actions:
        actions.zero_()

    root_state = base_env._robot.data.default_root_state[env_ids].clone()
    root_state[:, :3] += base_env._terrain.env_origins[env_ids]
    root_state[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=base_env.device)
    root_state[:, 7:] = 0.0
    base_env._robot.write_root_link_pose_to_sim(root_state[:, :7], env_ids)
    base_env._robot.write_root_com_velocity_to_sim(root_state[:, 7:], env_ids)

    joint_pos = base_env._robot.data.default_joint_pos[env_ids].clone()
    joint_vel = torch.zeros_like(base_env._robot.data.default_joint_vel[env_ids])
    base_env._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
    return obs


def summarize_lateral_deviation(name: str, lateral_history: torch.Tensor) -> dict[str, float]:
    """Summarize world-Y deviation from the initial line."""
    abs_lateral = torch.abs(lateral_history)
    return {
        "name": name,
        "mean_abs_m": abs_lateral.mean().item(),
        "rms_m": torch.sqrt(torch.mean(torch.square(lateral_history))).item(),
        "max_abs_m": abs_lateral.max().item(),
        "final_abs_m": abs_lateral[-1].mean().item(),
        "final_signed_m": lateral_history[-1].mean().item(),
    }


def print_lateral_summary(stats: dict[str, float]):
    print(
        "[RESULT] "
        f"{stats['name']:>5s} "
        f"mean_abs={stats['mean_abs_m']:.4f} m, "
        f"rms={stats['rms_m']:.4f} m, "
        f"max_abs={stats['max_abs_m']:.4f} m, "
        f"final_abs={stats['final_abs_m']:.4f} m, "
        f"final_signed={stats['final_signed_m']:.4f} m"
    )


def run_trial(env: RslRlVecEnvWrapper, ppo_runner: OnPolicyRunnerLoco, policy) -> dict[str, float]:
    obs = reset_robot_for_trial(env, ppo_runner)
    set_body_forward_command(env, args_cli.forward_speed)
    obs, _ = env.get_observations()

    base_env = env.unwrapped
    initial_pos_w = base_env._robot.data.root_pos_w.clone()
    lateral_history = []

    for timestep in range(args_cli.test_steps):
        if not simulation_app.is_running():
            break

        with torch.inference_mode():
            set_body_forward_command(env, args_cli.forward_speed)
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)

            set_body_forward_command(env, args_cli.forward_speed)
            if torch.any(dones):
                obs, _ = env.get_observations()

        lateral = base_env._robot.data.root_pos_w[:, 1] - initial_pos_w[:, 1]
        lateral_history.append(lateral.detach().clone())

        if args_cli.print_interval > 0 and timestep % args_cli.print_interval == 0:
            measured_vel_b = base_env._robot.data.root_com_lin_vel_b[0]
            measured_pos = base_env._robot.data.root_pos_w[0]
            cmd = torch.cat([base_env._cmd_lin_vel[0], base_env._cmd_ang_vel[0]]).detach()
            print(
                "[INFO] "
                f"step={timestep:05d} "
                f"cmd_b=[{cmd[0].item():.2f}, {cmd[1].item():.2f}, {cmd[2].item():.2f}] "
                f"vel_b=[{measured_vel_b[0].item():.2f}, {measured_vel_b[1].item():.2f}, "
                f"{measured_vel_b[2].item():.2f}] "
                f"y_dev={measured_pos[1].item() - initial_pos_w[0, 1].item():.3f} m"
            )

    if len(lateral_history) == 0:
        return summarize_lateral_deviation("body", torch.zeros(1, base_env.num_envs, device=base_env.device))
    return summarize_lateral_deviation("body", torch.stack(lateral_history, dim=0))


def main():
    env_cfg = parse_env_cfg(
        task_name,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    agent_cfg = cli_args.parse_rsl_rl_cfg(task_name, args_cli)

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    print(f"[INFO] Loading experiment from directory: {log_root_path}")

    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    log_dir = os.path.dirname(resume_path)
    print(f"[INFO] Checkpoint: {resume_path}")

    env_cfg.is_play_env = True
    env_cfg.follow_camera = False
    env_cfg.cmd_resample_interval = (1.0e9, 1.0e9)
    env_cfg.episode_length_s = max(env_cfg.episode_length_s, args_cli.test_steps * env_cfg.sim.dt * env_cfg.decimation + 1.0)
    env_cfg.terrain.terrain_type = "generator"
    env_cfg.terrain.terrain_generator.sub_terrains = {
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=1.0),
    }
    env_cfg.terrain.max_init_terrain_level = None
    env_cfg.viewer.eye = (6.0, -8.0, 4.5)
    env_cfg.viewer.lookat = (2.0, 0.0, 0.4)

    env = gym.make(task_name, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    # Keep debug markers created by is_play_env=True, then disable the env's per-step camera follow.
    env.unwrapped.cfg.is_play_env = False

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "fixed_forward"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording video.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    ppo_runner = OnPolicyRunnerLoco(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    adapt_policy_actor_to_checkpoint(ppo_runner, resume_path, agent_cfg)
    ppo_runner.load(resume_path, load_optimizer=False)
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    print("=" * 50)
    print("[INFO] Running body-frame forward command on flat terrain.")
    print("[INFO] Lateral deviation is measured as current world-y minus initial world-y.")
    print("=" * 50)
    stats = run_trial(env, ppo_runner, policy)
    print_lateral_summary(stats)

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
