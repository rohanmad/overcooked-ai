"""PPO algorithm. Tests: python -m pytest tests/test_ppo.py"""

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn


@dataclass
class PPOConfig:
    learning_rate: float = 3e-4
    gamma: float = 0.99         # how much future reward counts compared to immediate reward
    gae_lambda: float = 0.95
    clip_coef: float = 0.2      # how far one update may move the policy
    ent_coef: float = 0.01      # bonus for exploring
    vf_coef: float = 0.5        # weight of the critic's loss
    update_epochs: int = 4
    num_minibatches: int = 4
    max_grad_norm: float = 0.5


class RolloutBuffer:
    """One agent's steps in time order. dones[t] = 1.0 if the episode ended after step t."""

    def __init__(self):
        self.obs, self.actions, self.log_probs = [], [], []
        self.rewards, self.dones, self.values = [], [], []

    def add(self, obs, action, log_prob, reward, done, value):
        # TODO: Save one timestep of experience by appending each argument to its matching list above
        # (obs to self.obs, action to self.actions, and so on). get() below turns the lists into tensors later.
        raise NotImplementedError

    def get(self, gamma, gae_lambda):
        """Returns a dict of tensors: obs, actions, log_probs, advantages, returns."""
        f32 = lambda x: np.array(x, dtype=np.float32)
        advantages, returns = compute_gae(f32(self.rewards), f32(self.values), f32(self.dones), gamma, gae_lambda)
        return {
            "obs": torch.tensor(f32(self.obs)),
            "actions": torch.tensor(self.actions),
            "log_probs": torch.tensor(f32(self.log_probs)),
            "advantages": torch.tensor(advantages, dtype=torch.float32),
            "returns": torch.tensor(returns, dtype=torch.float32),
        }


def compute_gae(rewards, values, dones, gamma, gae_lambda):
    """Generalized Advantage Estimation. Arrays of length T -> (advantages, returns = advantages + values)."""
    # TODO: Work out, for every step, how much better the action turned out than the critic expected. That's
    # the "advantage", and PPO will make positive-advantage actions more likely. Loop backwards from t = T-1 down
    # to 0, keeping a running total `gae` that starts at 0:
    #     next_value = values[t + 1], or 0 on the last step
    #     not_done   = 1 - dones[t]   (stops one episode's future from leaking into the previous episode)
    #     delta      = rewards[t] + gamma * next_value * not_done - values[t]
    #     gae        = delta + gamma * gae_lambda * not_done * gae
    #     advantages[t] = gae
    # Return (advantages, advantages + values). The second array, the "returns", is what the critic learns to predict.
    raise NotImplementedError


def ppo_loss(model, batch, config):
    """
    Clipped PPO loss on one minibatch (normalize advantages first).
    Returns (loss, stats) with stats keys: policy_loss, value_loss, entropy, approx_kl.
    """
    # TODO: Measure how wrong the network is, as one number PyTorch can minimize. Call
    # model.get_action_and_value(batch["obs"], batch["actions"]) to get new log-probs, entropy and values for the
    # stored actions. ratio = exp(new_log_probs - batch["log_probs"]) says how much more (above 1) or less
    # (below 1) likely each action is now than when it was played. Normalize the advantages:
    # (adv - adv.mean()) / (adv.std() + 1e-8). The policy loss is the mean of
    # max(-adv * ratio, -adv * clamp(ratio, 1 - clip_coef, 1 + clip_coef)). This clipping is PPO's key idea: it
    # stops one update from changing the policy too much. The value loss is 0.5 * mean((new_values - returns)^2).
    # Total loss = policy_loss + vf_coef * value_loss - ent_coef * entropy.mean(). Also return a stats dict of
    # plain floats (.item()), including approx_kl = mean((ratio - 1) - log(ratio)), a measure of how far the
    # policy moved. Compute it inside torch.no_grad().
    raise NotImplementedError


def ppo_update(model, optimizer, batch, config):
    """config.update_epochs passes over shuffled minibatches, clipping grads to config.max_grad_norm.
    Returns the mean of each stat."""
    # TODO: Learn from the collected batch. Repeat config.update_epochs times: shuffle the indices
    # (torch.randperm(n)), split them into config.num_minibatches chunks (torch.chunk), and for each chunk take
    # those rows of every tensor in batch ({k: v[idx] for k, v in batch.items()}). Then: compute ppo_loss,
    # optimizer.zero_grad(), loss.backward(), nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm),
    # optimizer.step(). Keep every stats dict, and return one dict with the average of each stat.
    raise NotImplementedError
