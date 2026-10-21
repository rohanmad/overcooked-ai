import numpy as np
import torch

from model import ActorCritic
from ppo import PPOConfig, RolloutBuffer, compute_gae, ppo_loss, ppo_update

STATS = {"policy_loss", "value_loss", "entropy", "approx_kl"}


def test_buffer():
    buf = RolloutBuffer()
    for t in range(3):
        buf.add(np.zeros(4), 1, -0.5, 1.0, float(t == 2), 0.0)
    assert buf.get(0.9, 0.95)["obs"].shape == (3, 4)


def test_gae_lambda_one():
    adv, ret = compute_gae(np.ones(3), np.full(3, 0.5), np.array([0, 0, 1.0]), 0.9, 1.0)
    assert np.allclose(ret, [2.71, 1.9, 1.0]) and np.allclose(adv, ret - 0.5)


def test_gae_lambda_zero():
    adv, _ = compute_gae(np.array([1.0, 2.0]), np.array([0.5, 1.5]), np.array([0, 1.0]), 0.9, 0.0)
    assert np.allclose(adv, [1 + 0.9 * 1.5 - 0.5, 2 - 1.5])


def test_gae_stops_at_done():
    adv, _ = compute_gae(np.ones(2), np.array([0.0, 5.0]), np.array([1.0, 1.0]), 0.9, 0.95)
    assert np.allclose(adv, [1.0, -4.0])


def make_batch(model, n=256):
    obs = torch.randn(n, 4)
    with torch.no_grad():
        actions, log_probs, _, _ = model.get_action_and_value(obs)
    return {"obs": obs, "actions": actions, "log_probs": log_probs,
            "advantages": torch.randn(n), "returns": torch.randn(n)}


def test_ppo_loss():
    model = ActorCritic(4, 2)
    loss, stats = ppo_loss(model, make_batch(model), PPOConfig())
    assert loss.dim() == 0 and set(stats) == STATS
    assert abs(stats["approx_kl"]) < 1e-6  # unchanged policy


def test_ppo_update_learns():
    torch.manual_seed(0)
    model = ActorCritic(4, 2)
    batch = make_batch(model)
    batch["advantages"] = torch.where(batch["actions"] == 0, 1.0, -1.0)  # action 0 is always better
    prob_0 = lambda: torch.softmax(model(batch["obs"])[0], -1)[:, 0].mean().item()

    before = prob_0()
    stats = ppo_update(model, torch.optim.Adam(model.parameters(), lr=1e-2), batch, PPOConfig())
    assert set(stats) == STATS and prob_0() > before + 0.05
