from env_settings import *
from utils import run_single_episode

import gym
import os

import numpy as np



def get_demo(base_name, expert_policy, env_num, max_demo_length=np.inf):
    demonstrations = []

    env = gym.make('{}{}-v0'.format(base_name, env_num))
    layout = env.reset()

    t = 0
    while True:
        action = expert_policy(layout)
        demonstrations.append((layout, action))
        layout, reward, done, _ = env.step(action)
        t += 1
        if done or (t >= max_demo_length):
            if not reward > 0:
                print("WARNING: demo did not succeed!")
            break

    return demonstrations

def expert_nim_policy(layout):
    r1 = np.max(np.argwhere(layout == tpn.EMPTY)[:, 0])
    if layout[r1, 0] == tpn.TOKEN:
        c1 = 0
    elif layout[r1, 1] == tpn.TOKEN:
        c1 = 1
    else:
        r1 += 1
        c1 = 0
    return (r1, c1)

def expert_checkmate_tactic_policy(layout):
    if np.any(layout == checkmate_tactic.WHITE_QUEEN):
        return tuple(np.argwhere(layout == checkmate_tactic.WHITE_QUEEN)[0])

    black_king_pos = np.argwhere(layout == checkmate_tactic.BLACK_KING)[0]
    white_king_pos = np.argwhere(layout == checkmate_tactic.WHITE_KING)[0]

    return ((black_king_pos[0] + white_king_pos[0]) // 2, (black_king_pos[1] + white_king_pos[1]) // 2)

def expert_stf_policy(layout):
    r, c = np.argwhere(layout == stf.FALLING)[0]

    while True:
        if layout[r+1, c] in [stf.STATIC, stf.DRAWN]:
            break
        r += 1

    if layout[r, c-1] == stf.RED or layout[r, c+1] == stf.RED:
        return (r, c)

    r, c = np.argwhere(layout == stf.ADVANCE)[0]
    return (r, c)

def expert_ec_policy(layout):
    r, c = np.argwhere(layout == ec.TARGET)[0]
    ra, ca = np.argwhere(layout == ec.AGENT)[0]
    
    left_arrow = tuple(np.argwhere(layout == ec.LEFT_ARROW)[0])
    right_arrow = tuple(np.argwhere(layout == ec.RIGHT_ARROW)[0])
    up_arrow = tuple(np.argwhere(layout == ec.UP_ARROW)[0])
    down_arrow = tuple(np.argwhere(layout == ec.DOWN_ARROW)[0])

    # Top left corner
    if layout[r-1, c] == ec.WALL and layout[r, c-1] == ec.WALL:

        # Draw on right
        if layout[r, c+1] == ec.EMPTY:
            return (r, c+1)

        # Move to left
        if layout[ra, ca-1] == ec.EMPTY:
            return left_arrow

        # Move up
        return up_arrow

    # Top right corner
    if layout[r-1, c] == ec.WALL and layout[r, c+1] == ec.WALL:

        # Draw on left
        if layout[r, c-1] == ec.EMPTY:
            return (r, c-1)

        # Move to right
        if layout[ra, ca+1] == ec.EMPTY:
            return right_arrow

        # Move up
        return up_arrow

    # Bottom left corner
    if layout[r+1, c] == ec.WALL and layout[r, c-1] == ec.WALL:

        # Draw on right
        if layout[r, c+1] == ec.EMPTY:
            return (r, c+1)

        # Move to left
        if layout[ra, ca-1] == ec.EMPTY:
            return left_arrow

        # Move down
        return down_arrow

    # Bottom right corner
    if layout[r+1, c] == ec.WALL and layout[r, c+1] == ec.WALL:

        # Draw on left
        if layout[r, c-1] == ec.EMPTY:
            return (r, c-1)

        # Move to right
        if layout[ra, ca+1] == ec.EMPTY:
            return right_arrow

        # Move down
        return down_arrow

    # Wait
    return (0, 0)

def expert_rfts_policy(layout):
    agent_r, agent_c = np.argwhere(layout == rfts.AGENT)[0]
    star_r, star_c = np.argwhere(layout == rfts.STAR)[0]
    right_arrow = tuple(np.argwhere(layout == rfts.RIGHT_ARROW)[0])
    left_arrow = tuple(np.argwhere(layout == rfts.LEFT_ARROW)[0])

    height_to_star = agent_r - star_r

    # gonna climb up from the left
    if agent_c <= star_c:

        # move to the left more
        if abs(agent_c - star_c) < height_to_star:
            return left_arrow

        # stairs do not exist
        sr, sc = star_r+1, star_c
        while sc > agent_c:
            if sr >= layout.shape[0] - 2:
                break
            if layout[sr, sc] != rfts.DRAWN:
                return (sr, sc)
            sr += 1
            sc -= 1

        # move to the right
        return right_arrow

    # gonna climb up from the right
    else:
        # move to the right more
        if abs(agent_c - star_c) < height_to_star:
            return right_arrow

        # stairs do not exist
        sr, sc = star_r+1, star_c
        while sc < agent_c:
            if sr >= layout.shape[0] - 2:
                break
            if layout[sr, sc] != rfts.DRAWN:
                return (sr, sc)
            sr += 1
            sc += 1

        # move to the left
        return left_arrow

def get_expert_policy(env_name):
    return {
        'TwoPileNim' : expert_nim_policy,
        'CheckmateTactic' : expert_checkmate_tactic_policy,
        'StopTheFall' : expert_stf_policy,
        'Chase' : expert_ec_policy,
        'ReachForTheStar' : expert_rfts_policy
    }[env_name]


def get_demonstrations(env_name, demo_numbers=(1, 2, 3, 4), max_demo_length=np.inf):
    expert_policy = get_expert_policy(env_name)
    demonstrations = []

    for i in demo_numbers:
        demonstrations += get_demo(env_name, expert_policy, i, max_demo_length=max_demo_length)

    return [(np.array(l, dtype=object), a) for (l, a) in demonstrations]

def record_expert_demo(env_name, expert_policy, i, outdir, record_video=True):
    env = gym.make('{}{}-v0'.format(env_name, i))
    outpath = os.path.join(outdir, 'expert_demonstration_{}.mp4'.format(env.__class__.__name__))
    total_reward = run_single_episode(env, expert_policy, record_video=record_video, video_out_path=outpath)
    assert total_reward > 0


def record_expert_demos(env_name, demo_numbers=(1, 2, 3, 4), outdir='/tmp', record_video=True):
    expert_policy = get_expert_policy(env_name)

    for i in demo_numbers:
        print("Recording expert demo", i)
        record_expert_demo(env_name, expert_policy, i, outdir, record_video=record_video)


if __name__ == "__main__":
    record_expert_demos('TwoPileNim', demo_numbers=(0,1,2))
    record_expert_demos('CheckmateTactic', demo_numbers=(0,1,2))
    record_expert_demos('Chase', demo_numbers=(0,1,2))
    record_expert_demos('StopTheFall', demo_numbers=(0,1,2))
    record_expert_demos('ReachForTheStar', demo_numbers=(0,1,2))
