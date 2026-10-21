"""
Smoke test: run one full Overcooked episode on cramped_room with two random agents.

Run with:  python hello_overcooked.py
"""

import random

from overcooked_ai_py.mdp.actions import Action
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld

LAYOUT = "cramped_room"
HORIZON = 400  # standard episode length in the Overcooked literature
SEED = 0


def main():
    random.seed(SEED)

    # The MDP holds the game rules + layout; the Env adds the episode loop (horizon, reset).
    mdp = OvercookedGridworld.from_layout_name(LAYOUT)
    env = OvercookedEnv.from_mdp(mdp, horizon=HORIZON, info_level=0)
    env.reset()

    print(f"Layout: {LAYOUT} | actions per agent: {Action.NUM_ACTIONS}")
    print("Start state:")
    print(env)  # OvercookedEnv.__repr__ renders the kitchen as text

    total_sparse = 0  # soup delivery reward (20 per soup), shared by both agents
    total_shaped = 0  # dense "helper" reward (e.g. putting onion in pot); not used here
    done = False
    while not done:
        # Action.ALL_ACTIONS = [north, south, east, west, stay, interact]
        joint_action = (random.choice(Action.ALL_ACTIONS), random.choice(Action.ALL_ACTIONS))
        _, sparse_reward, done, info = env.step(joint_action)
        total_sparse += sparse_reward
        total_shaped += sum(info["shaped_r_by_agent"])

    print(f"Final state (timestep {env.state.timestep}):")
    print(env)
    print(f"Total soup (sparse) reward: {total_sparse}")
    print(f"Total shaped reward (both agents): {total_shaped}")


if __name__ == "__main__":
    main()
