"""Tactile gripper sorting environments with Elfin robot."""

import gymnasium as gym

# Import environment and configuration
from .elfin_lift_env import ElfinTactileLiftEnvCfg, ElfinTactileLiftEnv

# Register Gym environments with Isaac Lab
gym.register(
    id="Isaac-ElfinTactile-Direct-v0",
    entry_point="testTask_env.tasks.my_elfin_lift.elfin_lift_env:ElfinTactileLiftEnv",
    disable_env_checker=True,
    kwargs={
        "cfg": ElfinTactileLiftEnvCfg,
        "render_mode": None,
    },
)

gym.register(
    id="Isaac-ElfinTactile-Direct-Play-v0", 
    entry_point="testTask_env.tasks.my_elfin_lift.elfin_lift_env:ElfinTactileLiftEnv",
    disable_env_checker=True,
    kwargs={
        "cfg": ElfinTactileLiftEnvCfg,
        "render_mode": "rgb_array",
    },
)