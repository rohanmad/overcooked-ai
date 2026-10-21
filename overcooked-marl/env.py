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
        # TODO: Start a new game and return what each chef "sees". The game lives in self.base_env, so call
        # self.base_env.reset() first. Then turn the current game state (self.base_env.state) into numbers with
        # self.base_env.featurize_state_mdp(state). That returns a list of two arrays, one per chef, each
        # describing the kitchen from that chef's point of view. Return them as a dict mapping "agent_0" and
        # "agent_1" to their arrays, converted to np.float32. step() needs the same thing, so a small helper
        # method is a good idea.
        raise NotImplementedError

    def step(self, actions):
        """
        actions: {agent: int in 0..5}
        Returns (obs, rewards, done, info):
            rewards: {agent: soup reward + shaped_reward_coef * that agent's shaped reward}
            done:    True when the episode hits the horizon
            info:    {"sparse_reward": float}
        """
        # TODO: Move the game forward one timestep. Our agents pick actions as integers 0-5, but the game wants
        # its own action objects, so convert each one with Action.INDEX_TO_ACTION[i]. Then call
        # self.base_env.step(joint_action), where joint_action is a tuple (agent_0's action, agent_1's action).
        # It returns (state, sparse_reward, done, info). sparse_reward is the soup reward (20 per delivered soup),
        # shared by both chefs because this is a team game. info["shaped_r_by_agent"] is a list of two small bonus
        # rewards for useful steps like putting an onion in the pot. Each chef's reward is the soup reward plus
        # self.shaped_reward_coef times its own bonus. Return the new observations, the rewards dict, done, and
        # {"sparse_reward": sparse_reward}.
        raise NotImplementedError
