import argparse
import logging
import time
import gym
import numpy as np
import pickle
import lbforaging
from lbforaging.agents.expert_policy import get_accessible_obs
logger = logging.getLogger(__name__)


def _game_loop(env, render, ppl, n_agents):
    """
    """
    obs = env.reset()
    info = env.get_player_pos_info()

    done = False

    if render:
        env.render()
        time.sleep(0.5)

    while not done:
        actions = []
        for i in range(n_agents):
            actions.append(ppl[i](get_accessible_obs(obs[0])[:2], info['player_pos'][i]))

        obs, nreward, ndone, info = env.step(actions)

        if sum(nreward) > 0:
            print(nreward)

        if render:
            env.render()
            time.sleep(0.5)

        done = np.all(ndone)
    print(env.players[0].score, env.players[1].score)


def main(env_name, game_count=1, render=False):
    env = gym.make(env_name)
    obs = env.reset()
    n_agents = env.n_agents
    ppl_file = f'Foraging-grid-5x5-2p-2f-v2_ppl.pkl'
    print(f'load policies from {ppl_file}')
    with open(ppl_file, 'rb') as f:
        ppl = pickle.load(f)

    for episode in range(game_count):
        _game_loop(env, render, ppl, n_agents)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Play the level foraging game.")

    parser.add_argument("--render", action="store_true")
    parser.add_argument(
        "--times", type=int, default=1, help="How many times to run the game"
    )
    parser.add_argument(
        "--env", type=str, default="Foraging-grid-10x10-5p-5f-v2", help="The env to run"
    )

    args = parser.parse_args()
    main(args.env, args.times, args.render)
