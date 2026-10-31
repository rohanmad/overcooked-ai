# Roadmap

Goal: PPO agents that learn to cook and deliver soup on `cramped_room`, demoed at the showcase on **Dec 2**.
Run all commands from `overcooked-marl/` with `.venv` active.

## Week 1 (Oct 26 – Nov 1): Setup and first code
- [ ] Everyone: setup done, `python hello_overcooked.py` runs
- [ ] `env.py`: `reset()` (`python -m pytest tests/test_env.py -k reset`)
- [ ] `model.py`: `__init__`, `forward` (`python -m pytest tests/test_model.py -k forward`)
- [ ] `ppo.py`: `RolloutBuffer.add()`
- [ ] Read and work through `compute_gae` on paper

## Week 2 (Nov 2 – 8): The game and the brain
- [ ] `env.py`: `step()` (`python -m pytest tests/test_env.py`)
- [ ] `model.py`: `get_action_and_value()` (`python -m pytest tests/test_model.py`)
- [ ] `ppo.py`: `compute_gae()` (`python -m pytest tests/test_ppo.py -k "gae or buffer"`)
- [ ] `python watch.py` shows untrained chefs moving in a window

## Week 3 (Nov 9 – 15): Learning code
- [ ] `ppo.py`: `ppo_loss()`, `ppo_update()` (`python -m pytest tests/test_ppo.py`)
- [ ] `train.py`: `collect_rollouts()` (`python -m pytest tests/test_train.py`)
- [ ] All tests pass (`python -m pytest`)
- [ ] Short trial run: set `TOTAL_STEPS = 100_000` in `train.py`, run `python train.py` to the end

## Week 4 (Nov 16 – 22): Make it learn
- [ ] Full training runs overnight (`TOTAL_STEPS = 1_000_000`)
- [ ] Shaped reward rises, then soup reward rises above 0
- [ ] If flat: change one setting at a time (more steps, `ent_coef = 0.02`, `learning_rate = 1e-4`), log each run
- [ ] **Nov 22: training must work.** After this, only demo prep

## Week 5 (Nov 23 – 29): Demo material (Thanksgiving week)
- [ ] Final training run, ideally 2–3 seeds
- [ ] Screen recording of `python watch.py` with the trained model
- [ ] Chart of soup reward over training
- [ ] Draft slides or poster

## Showcase (Nov 30 – Dec 2)
- [ ] Finish slides or poster
- [ ] Rehearse twice
- [ ] Backup video ready in case the live demo fails
- [ ] **Dec 2: showcase**
