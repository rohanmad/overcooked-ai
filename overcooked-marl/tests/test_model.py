import torch

from model import ActorCritic

B, OBS_DIM, N_ACTIONS = 5, 8, 6


def test_forward_shapes():
    logits, value = ActorCritic(OBS_DIM, N_ACTIONS)(torch.zeros(B, OBS_DIM))
    assert logits.shape == (B, N_ACTIONS) and value.shape == (B,)


def test_get_action_and_value():
    model = ActorCritic(OBS_DIM, N_ACTIONS)
    obs = torch.randn(B, OBS_DIM)
    outputs = model.get_action_and_value(obs)
    assert all(x.shape == (B,) for x in outputs)

    actions = torch.arange(B)
    _, log_prob, _, _ = model.get_action_and_value(obs, actions)
    expected = torch.log_softmax(model(obs)[0], dim=-1)[torch.arange(B), actions]
    assert torch.allclose(log_prob, expected, atol=1e-6)
