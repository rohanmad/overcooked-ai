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
    buffers = {a: RolloutBuffer() for a in AGENTS}
    total_soup = 0.0

    for _ in range(n_episodes):
        obs = env.reset()
        done = False
        while not done:
            actions, log_probs, values = {}, {}, {}
            with torch.no_grad():
                for a in AGENTS:
                    action, log_prob, _, value = model.get_action_and_value(torch.as_tensor(obs[a]).unsqueeze(0))
                    actions[a], log_probs[a], values[a] = action.item(), log_prob.item(), value.item()

            next_obs, rewards, done, info = env.step(actions)
            for a in AGENTS:
                buffers[a].add(obs[a], actions[a], log_probs[a], rewards[a], float(done), values[a])
            total_soup += info["sparse_reward"]
            obs = next_obs

    data = [buffers[a].get(config.gamma, config.gae_lambda) for a in AGENTS]
    batch = {k: torch.cat([d[k] for d in data]) for k in data[0]}
    return batch, total_soup / n_episodes


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
