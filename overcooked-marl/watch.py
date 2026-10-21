"""Watch the agents play one episode in a window. Run: python watch.py [--model model.pt] [--fps 10]"""

import argparse
from pathlib import Path

import pygame
import torch
from overcooked_ai_py.visualization.state_visualizer import StateVisualizer

from env import AGENTS, N_ACTIONS, OBS_DIM, OvercookedMultiAgentEnv
from model import ActorCritic
from train import LAYOUT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="model.pt")
    parser.add_argument("--fps", type=int, default=10)
    args = parser.parse_args()

    model = ActorCritic(OBS_DIM, N_ACTIONS)
    if Path(args.model).exists():
        model.load_state_dict(torch.load(args.model))
    else:
        print(f"{args.model} not found, using an untrained network")

    env = OvercookedMultiAgentEnv(LAYOUT, shaped_reward_coef=0.0)
    visualizer = StateVisualizer()
    pygame.init()
    clock, window = pygame.time.Clock(), None
    obs, done, score = env.reset(), False, 0.0

    while not any(e.type == pygame.QUIT for e in pygame.event.get()):  # close the window to quit
        state = env.base_env.state
        frame = visualizer.render_state(state, env.mdp.terrain_mtx, StateVisualizer.default_hud_data(state, score=score))
        window = window or pygame.display.set_mode(frame.get_size())
        window.blit(frame, (0, 0))
        pygame.display.flip()

        if not done:
            with torch.no_grad():
                actions = {a: model.get_action_and_value(torch.as_tensor(obs[a]).unsqueeze(0))[0].item() for a in AGENTS}
            obs, _, done, info = env.step(actions)
            score += info["sparse_reward"]
        clock.tick(args.fps)
    pygame.quit()


if __name__ == "__main__":
    main()
