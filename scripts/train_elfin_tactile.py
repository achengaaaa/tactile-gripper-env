#!/usr/bin/env python3
"""Train tactile gripper sorting policy with Elfin robot."""

import argparse
import gymnasium as gym
from omni.isaac.lab.app import AppLauncher

# Parse command line arguments
parser = argparse.ArgumentParser(description="Train tactile gripper sorting policy with Elfin robot.")
parser.add_argument("--num_envs", type=int, default=64, help="Number of environments to spawn.")
parser.add_argument("--headless", action="store_true", default=False, help="Force display off at all times.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of recorded videos.")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings.")
parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible results.")

# Parse arguments
args_cli = parser.parse_args()

# Launch Isaac Sim
config = {"headless": args_cli.headless}
app_launcher = AppLauncher(config)
simulation_app = app_launcher.app

"""Rest of the script after Isaac Sim app is initialized."""

import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False

from omni.isaac.lab.envs import ManagerBasedRLEnvCfg
from omni.isaac.lab.utils.dict import print_dict

import omni.isaac.lab_tasks  # noqa: F401
from omni.isaac.lab_tasks.utils import parse_env_cfg

# Import custom environment configuration
from testTask_env.tasks.my_elfin_lift.elfin_lift_env import ElfinTactileLiftEnvCfg


def main():
    """Main function."""
    # Parse configuration
    env_cfg: ElfinTactileLiftEnvCfg = ElfinTactileLiftEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    # Set video recording if enabled
    if args_cli.video:
        env_cfg.viewer.record_enabled = True
        env_cfg.viewer.record_length = args_cli.video_length
        env_cfg.viewer.record_interval = args_cli.video_interval

    # Print environment configuration
    print("Environment Configuration:")
    print_dict(env_cfg.to_dict(), nesting=4)
    print(f"Number of environments: {env_cfg.scene.num_envs}")
    print(f"Environment device: {env_cfg.sim.device}")

    # Create environment
    print("Creating environment...")
    env = gym.make("Isaac-ElfinTactile-Direct-v0", cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    print(f"Environment created successfully!")
    print(f"Environment info:")
    print(f"  Observation space: {env.observation_space}")
    print(f"  Action space: {env.action_space}")

    # Test random actions
    print("\nTesting environment with random actions...")
    
    # Reset environment
    obs, _ = env.reset(seed=args_cli.seed)
    print(f"Initial observation shape: {obs['policy'].shape}")
    
    # Run for a few episodes
    episode_count = 0
    step_count = 0
    
    try:
        for i in range(1000):  # Run for 1000 steps
            # Sample random actions
            actions = env.action_space.sample()
            
            # Step environment
            obs, rewards, terminated, truncated, info = env.step(actions)
            
            step_count += 1
            
            # Print progress every 100 steps
            if (i + 1) % 100 == 0:
                print(f"Step {step_count}: Running successfully")
                try:
                    print(f"  Current rewards: {rewards}")
                except:
                    print(f"  Current rewards: [data]")
    
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    
    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Close environment
        print("\nClosing environment...")
        env.close()
        print("Environment closed successfully.")


if __name__ == "__main__":
    try:
        main()
    finally:
        # Close simulation app
        simulation_app.close()
