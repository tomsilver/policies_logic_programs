import numpy as np
from expert_demonstrations import get_demo
import pickle
import pipeline
import argparse
import os

parser = argparse.ArgumentParser(description='Train the logical program policies on the ForagingEnv')
parser.add_argument('n_demos', type=int, default=10,
                    help='an integer for the accumulator')
parser.add_argument('num_programs', type=int, default=10000,
                    help='the number of feature detector programs')
parser.add_argument('num_dt', type=int, default=5,
                    help='the number of feature detector programs')
parser.add_argument('max_num_particles', type=int, default=50,
                    help='the number of feature detector programs')
parser.add_argument('gen_prog_step_size', type=int, default=1,
                    help='the number of feature detector programs')


args = parser.parse_args()

policy_file_name = f'policy-{args.n_demos}-{args.num_programs}-{args.num_dt}-{args.max_num_particles}-{args.gen_prog_step_size}.pkl'

if os.path.exists(policy_file_name):
    with open(policy_file_name, 'rb') as inp:
        policy = pickle.load(inp)
    print(f'loaded policy "{policy_file_name}" from pickle.')
else:
    policy = pipeline.train("ForagingEnv", range(args.n_demos), args.gen_prog_step_size,
                            args.num_programs, args.num_dt, args.max_num_particles)
    with open(policy_file_name, 'wb') as outp:
        pickle.dump(policy, outp, pickle.HIGHEST_PROTOCOL)

for d in range(0, 50, 10):
    demo = get_demo('ForagingEnv', None, d, max_demo_length=np.inf)

    for obs, action, pos in demo:
        print(policy(obs, pos), action)
