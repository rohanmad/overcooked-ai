"""Self-play PPO training on Overcooked. Run: python train.py   Tests: python -m pytest tests/test_train.py"""

import torch

from env import AGENTS, N_ACTIONS, OBS_DIM, OvercookedMultiAgentEnv
from model import ActorCritic
from ppo import PPOConfig, RolloutBuffer, ppo_update

LAYOUT = "cramped_room"
TOTAL_STEPS = 1_000_000
EPISODES_PER_UPDATE = 5
SHAPING_ANNEAL_STEPS = 500_000  # shaped reward weight decays 1 -> 0 over this many steps


def collect_rollouts(env, model, n_episodes, config):
    """
    Plays n_episodes with `model` controlling both agents.
    Returns (batch, mean soup reward per episode). batch: both agents' buffer.get() dicts concatenated.
    """
    # TODO: Play games with the current network and record everything for PPO. "Self-play" means one network
    # controls both chefs. Make one RolloutBuffer per agent ({a: RolloutBuffer() for a in AGENTS}); each chef's
    # experience has to stay in time order for compute_gae. For each episode, call obs = env.reset() and loop
    # until done. At each step, for each agent, call model.get_action_and_value on that agent's observation
    # (wrap it with torch.as_tensor(obs[a]).unsqueeze(0) to make a batch of 1, inside torch.no_grad()) and keep
    # the action, log-prob and value as plain numbers with .item(). Step the env with both actions, then add
    # each agent's (obs, action, log_prob, reward, float(done), value) to its buffer. Add up info["sparse_reward"]
    # over all episodes. At the end, call .get(config.gamma, config.gae_lambda) on both buffers and join the two
    # dicts with torch.cat for each key. Return (batch, total soup reward / n_episodes).
    raise NotImplementedError


def main():
    torch.manual_seed(0)
    config = PPOConfig()
    env = OvercookedMultiAgentEnv(LAYOUT)
    model = ActorCritic(OBS_DIM, N_ACTIONS)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    steps_per_update = EPISODES_PER_UPDATE * env.base_env.horizon
    for update in range(TOTAL_STEPS // steps_per_update):
        steps = update * steps_per_update
        env.shaped_reward_coef = max(0.0, 1 - steps / SHAPING_ANNEAL_STEPS)
        batch, soup_reward = collect_rollouts(env, model, EPISODES_PER_UPDATE, config)
        stats = ppo_update(model, optimizer, batch, config)
        print(f"steps {steps:8d} | soup reward {soup_reward:6.1f} | entropy {stats['entropy']:.2f}")

    torch.save(model.state_dict(), "model.pt")


if __name__ == "__main__":
    main()
