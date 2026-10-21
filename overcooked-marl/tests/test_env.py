import numpy as np

from env import AGENTS, OBS_DIM, OvercookedMultiAgentEnv

UP, STAY = 0, 4


def test_reset():
    obs = OvercookedMultiAgentEnv(horizon=10).reset()
    assert set(obs) == set(AGENTS)
    assert all(o.shape == (OBS_DIM,) and o.dtype == np.float32 for o in obs.values())


def test_done_at_horizon():
    env = OvercookedMultiAgentEnv(horizon=10)
    env.reset()
    dones = [env.step({a: STAY for a in AGENTS})[2] for _ in range(10)]
    assert dones == [False] * 9 + [True]


def test_action_mapping():
    env = OvercookedMultiAgentEnv(horizon=10)
    env.reset()
    env.step({"agent_0": UP, "agent_1": STAY})
    assert env.base_env.state.players[0].position == (1, 1)
    assert env.base_env.state.players[1].position == (3, 1)


def test_no_shaping_means_sparse_reward():
    env = OvercookedMultiAgentEnv(horizon=50, shaped_reward_coef=0.0)
    env.reset()
    rng = np.random.default_rng(0)
    for _ in range(50):
        _, rewards, _, info = env.step({a: int(rng.integers(6)) for a in AGENTS})
        assert all(r == info["sparse_reward"] for r in rewards.values())
