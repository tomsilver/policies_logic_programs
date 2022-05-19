import numpy as np
from expert_demonstrations import get_demo
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
                    help='the max number of particles for the decision tree')
parser.add_argument('--gen_prog_step_size', type=int, default=1,
                    help='the step size of the program generation loop')
parser.add_argument('--env_name', type=str, default="Foraging-grid-6x6-2p-3f-v2",
                    help='the step size of the program generation loop')

args = parser.parse_args()
print('Train policy with args:', ' '.join(f'{k}={v}' for k, v in vars(args).items()))

policy_file_name = f'policy-{args.n_demos}-{args.num_programs}-{args.num_dt}-{args.max_num_particles}-{args.gen_prog_step_size}.pkl'

policy = pipeline.train(args.env_name, range(args.n_demos), args.gen_prog_step_size,
                        args.num_programs, args.num_dt, args.max_num_particles)

for d in range(10):
    demo = get_demo(args.env_name, expert_foraging_policy, d, max_demo_length=np.inf)
    print('DEMO', d)
    for obs, action, pos in demo:
        print(policy(obs, pos), action)
