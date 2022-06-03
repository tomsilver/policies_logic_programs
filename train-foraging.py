import numpy as np
from expert_demonstrations import get_demo, run_foraging_policy
import pickle
import pipeline
import argparse
import os
from lbforaging.agents.expert_policy import expert_policy as expert_foraging_policy

parser = argparse.ArgumentParser(description='Train the logical program policies on the ForagingEnv')
parser.add_argument('--n_demos', type=int, default=10,
                    help='an integer for the accumulator')
parser.add_argument('--num_programs', type=int, default=10000,
                    help='the number of feature detector programs')
parser.add_argument('--num_dt', type=int, default=5,
                    help='the number of decision trees')
parser.add_argument('--max_num_particles', type=int, default=25,
                    help='the max number of particles for the plp')
parser.add_argument('--gen_prog_step_size', type=int, default=1,
                    help='the step size of the program generation loop')
parser.add_argument('--env_name', type=str, default="Foraging-grid-6x6-2p-3f-v2",
                    help='the step size of the program generation loop')

args = parser.parse_args()
print('Train policy with args:', ' '.join(f'{k}={v}' for k, v in vars(args).items()))

policy = pipeline.train(args.env_name, range(args.n_demos), args.gen_prog_step_size,
                        args.num_programs, args.num_dt, args.max_num_particles)

envs = [args.env_name, 'Foraging-grid-8x8-2p-4f-v2', 'Foraging-grid-12x12-2p-4f-v2', 'Foraging-grid-14x14-2p-5f-v2',
        'Foraging-grid-16x16-2p-6f-v2', 'Foraging-grid-18x18-2p-7f-v2', 'Foraging-grid-20x20-2p-8f-v2']
for i, e in enumerate(envs):
    rewards = []
    for d in range(1000):
        rewards.append(run_foraging_policy(e, policy, render=False, max_demo_length=20*(i+1)))
    print(f'env: {e}, avg. reward after run {d+1}: ' + str(np.array(rewards).mean()))

# 'Foraging-grid-8x8-2p-4f-v2'
# for d in range(10):
#     demo = get_demo(args.env_name, expert_foraging_policy, d, max_demo_length=np.inf)
#     print('DEMO', d)
#     for obs, action, pos in demo:
#         print(policy(obs, pos), action)
