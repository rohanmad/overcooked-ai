# overcooked-ai

Train two PPO agents to cooperate in Overcooked (self-play on `cramped_room`).

## Setup

```bash
uv venv --python 3.11 .venv && source .venv/bin/activate
uv pip install -r overcooked-marl/requirements.txt
python hello_overcooked.py      # should print a kitchen and a soup reward of 0
cd overcooked-marl
```

Activate `.venv` in every new terminal. Ignore the "Gym has been unmaintained" warning.

## What to implement

Fill in each `# TODO` in `overcooked-marl/`, in this order. The TODO comments explain what to do.

| File | Implement | Owner | Check with |
|---|---|---|---|
| `env.py` | `reset`, `step` | A | `python -m pytest tests/test_env.py` |
| `model.py` | `ActorCritic` | B | `python -m pytest tests/test_model.py` |
| `ppo.py` | `RolloutBuffer.add`, `compute_gae`, `ppo_loss`, `ppo_update` | C (+B) | `python -m pytest tests/test_ppo.py` |
| `train.py` | `collect_rollouts` | A | `python -m pytest tests/test_train.py` |

Use `python -m pytest`, not `pytest`, so the tests run with the project's Python.

## Run

```bash
python train.py     # train; prints soup reward, saves model.pt
python watch.py     # watch model.pt play (works untrained once env.py and model.py are done)
```

## Plan

| Week | Goal | Done when |
|---|---|---|
| 1 | Setup, RL basics, git workflow; start `env.py` and `model.py` | Everyone runs `hello_overcooked.py` |
| 2 | Finish `env.py` and `model.py`; `RolloutBuffer` and `compute_gae` | `watch.py` shows chefs moving; env, model and GAE tests pass |
| 3 | Finish `ppo.py` and `collect_rollouts`; first training run; debug together | All tests pass, `train.py` runs to the end |
| 4 | Get it learning: overnight runs, tune one setting at a time, log each run | Soup reward reliably above 0 |
| 5 | Final runs (2–3 seeds), demo recording, reward chart, write-up | Demo, chart, report |

