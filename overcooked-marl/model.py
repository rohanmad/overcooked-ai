"""Actor-critic network. Tests: python -m pytest tests/test_model.py"""

import torch.nn as nn
from torch.distributions import Categorical


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden_size=64):
        super().__init__()
        # TODO: Build two small neural networks. The "actor" decides what to do: it takes an observation and
        # outputs one score (a "logit") per action, where a higher score means the action is more likely to be
        # picked. The "critic" judges how good the current situation is: it takes the same observation and outputs
        # a single number, its guess of how much reward is coming. Build each with nn.Sequential as
        # Linear(obs_dim, hidden_size) -> Tanh -> Linear(hidden_size, hidden_size) -> Tanh -> Linear(hidden_size, out),
        # with out = n_actions for self.actor and out = 1 for self.critic.
        raise NotImplementedError

    def forward(self, obs):
        """obs (B, obs_dim) -> logits (B, n_actions), value (B,)"""
        # TODO: Run obs through both networks and return (logits, value). obs holds B observations at once
        # (a "batch"). The critic's output has shape (B, 1); drop the extra dimension with .squeeze(-1) so it's (B,).
        raise NotImplementedError

    def get_action_and_value(self, obs, action=None):
        """Samples an action if none is given. Returns action, log_prob, entropy, value, each (B,)."""
        # TODO: This is the one function PPO calls on the network. Get (logits, value) from self(obs), then build a
        # probability distribution with dist = Categorical(logits=logits). While playing, action is None: pick one
        # with dist.sample(). While learning, PPO passes in actions taken earlier, so use those instead. Return
        # the action, dist.log_prob(action) (log of how likely the policy thinks that action is), dist.entropy()
        # (how spread out the probabilities are, i.e. how much it's still exploring), and value.
        raise NotImplementedError
