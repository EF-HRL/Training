"""Collect five-frame tracking-error examples using frozen navigation and locomotion."""
import argparse
import os

from isaaclab.app import AppLauncher
import cli_args

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--disable_fabric", action="store_true")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--num_samples", type=int, default=8_192_000)
parser.add_argument("--filename", default="loco_error_dataset_nav_guided.h5")
parser.add_argument("--overwrite", action="store_true")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.num_samples < 1:
    parser.error("--num_samples must be positive")
app = AppLauncher(args).app

import go2_lidar.tasks  # registers the navigation environment
import gymnasium as gym
import torch
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
from rsl_rl.runners import OnPolicyRunner
from go2_lidar.utils.hdf5_data_collector import HDF5DatasetWriter_General


def main():
    task = "Unitree-Go2-Navigation"
    cfg = parse_env_cfg(task, device=args.device, num_envs=args.num_envs, use_fabric=not args.disable_fabric)
    # Keep the navigation terrain generator, randomized patch goals and friction ranges.
    cfg.is_play_env = False
    cfg.wait_for_key = False
    cfg.events.push_robot = None
    cfg.set_raycaster_measure_pattern("11x")
    agent_cfg = cli_args.parse_rsl_rl_cfg(task, args)
    run = args.load_run or "nav_1"
    path = get_checkpoint_path(os.path.abspath(os.path.join("logs/rsl_rl", agent_cfg.experiment_name)),
                               run, agent_cfg.load_checkpoint)
    env = RslRlVecEnvWrapper(gym.make(task, cfg=cfg), clip_actions=agent_cfg.clip_actions)
    writer = None
    try:
        base = env.unwrapped
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), device=base.device, use_ray=True)
        runner.load(path, load_optimizer=False)
        policy = runner.get_inference_policy(device=base.device)
        history = torch.zeros(base.num_envs, 5, 12, device=base.device)
        age = torch.zeros(base.num_envs, dtype=torch.long, device=base.device)
        writer = HDF5DatasetWriter_General(filename=args.filename, input_dim=12, output_dim=3,
                                            history_frames=5, num_total=args.num_samples, overwrite=args.overwrite)
        obs, _ = env.get_observations()
        with torch.inference_mode():
            while app.is_running() and writer.idx < args.num_samples:
                nav = policy(obs)
                data = base._robot.data
                v, omega = data.root_com_lin_vel_b, data.root_com_ang_vel_b
                frame = torch.cat((data.projected_gravity_b, v, omega, nav), dim=1)
                history = torch.roll(history, -1, dims=1)
                history[:, -1] = frame
                error = nav - torch.cat((v[:, :2], omega[:, 2:3]), dim=1)
                age += 1
                ids = (age >= 5).nonzero().flatten()[:args.num_samples - writer.idx]
                if ids.numel():
                    writer.add_data(ids.numel(), history[ids].cpu().numpy(), error[ids].cpu().numpy())
                obs, _, dones, _ = env.step(nav)
                runner.alg.policy.reset(dones)
                reset = dones.bool()
                age[reset] = 0
                history[reset] = 0
                reset_ids = reset.nonzero().flatten().cpu()
                if reset_ids.numel():
                    base._reset_physx_materials(reset_ids)
    finally:
        if writer is not None:
            writer.close()
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        app.close()
