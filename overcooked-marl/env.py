"""Two-agent Overcooked environment. Tests: python -m pytest tests/test_env.py"""

import numpy as np
from overcooked_ai_py.mdp.actions import Action
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld

AGENTS = ["agent_0", "agent_1"]
OBS_DIM = 96
N_ACTIONS = 6


class OvercookedMultiAgentEnv:
    def __init__(self, layout="cramped_room", horizon=400, shaped_reward_coef=1.0):
        self.shaped_reward_coef = shaped_reward_coef
        self.mdp = OvercookedGridworld.from_layout_name(layout)
        self.base_env = OvercookedEnv.from_mdp(self.mdp, horizon=horizon, info_level=0)

    def reset(self):
        """Returns obs: {agent: float32 array of shape (OBS_DIM,)}."""
        self.base_env.reset()
        return self._get_obs()

    def _get_obs(self):
        arrays = self.base_env.featurize_state_mdp(self.base_env.state)
        return {agent: arr.astype(np.float32) for agent, arr in zip (AGENTS, arrays)}

    def step(self, actions):
        """
        actions: {agent: int in 0..5}
        Returns (obs, rewards, done, info):
            rewards: {agent: soup reward + shaped_reward_coef * that agent's shaped reward}
            done:    True when the episode hits the horizon
            info:    {"sparse_reward": float}
        """
        joint_action = (
            Action.INDEX_TO_ACTION[actions["agent_0"]],
            Action.INDEX_TO_ACTION[actions["agent_1"]],
        )
        state, sparse_reward, done, info = self.base_env.step(joint_action)
        shaped = info["shaped_r_by_agent"]
        rewards = {
            agent: sparse_reward + self.shaped_reward_coef * bonus
            for agent, bonus in zip(AGENTS, shaped)
        }
        return self._get_obs(), rewards, done, {"sparse_reward": sparse_reward}
