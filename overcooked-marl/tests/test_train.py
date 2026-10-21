from env import N_ACTIONS, OBS_DIM, OvercookedMultiAgentEnv
from model import ActorCritic
from ppo import PPOConfig
from train import collect_rollouts


def test_collect_rollouts():
    env = OvercookedMultiAgentEnv(horizon=20)
    batch, soup_reward = collect_rollouts(env, ActorCritic(OBS_DIM, N_ACTIONS), 2, PPOConfig())
    assert len(batch["obs"]) == 2 * 20 * 2  # episodes x steps x agents
    assert soup_reward >= 0
