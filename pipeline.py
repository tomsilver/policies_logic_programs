from operator import index
from cache_utils import manage_cache
from dsl import *
from env_settings import *
from grammar_utils import generate_programs
from dt_utils import extract_plp_from_dt
from expert_demonstrations import get_demonstrations, run_foraging_policy
from policy import StateActionProgram, PLPPolicy
from utils import run_single_episode

from collections import defaultdict
from functools import partial
from sklearn.tree import DecisionTreeClassifier
from scipy.special import logsumexp
from scipy.sparse import csr_matrix, lil_matrix, vstack
import random
import gym
import multiprocessing
import numpy as np
import time
import os

from tqdm import tqdm

from lbforaging.foraging.environment import Action as lba

cache_dir = 'cache'


# @manage_cache(cache_dir, ['.pkl', '.pkl'])
def get_program_set(base_class_name, num_programs):
    """
    Enumerate all programs up to a certain iteration.

    Parameters
    ----------
    base_class_name : str
    num_programs : int

    Returns
    -------
    programs : [ StateActionProgram ]
        A list of programs in enumeration order.
    program_prior_log_probs : [ float ]
        Log probabilities for each program.
    """
    object_types = get_object_types(base_class_name)
    grammar = create_grammar(object_types)

    program_generator = generate_programs(grammar)
    programs = []
    program_prior_log_probs = []

    print("Generating {} programs".format(num_programs))
    for _ in range(num_programs):
        program, lp = next(program_generator)
        programs.append(program)
        program_prior_log_probs.append(lp)
    print("\nDone.")

    return programs, program_prior_log_probs


def is_fruit_direction_compatible_with_action(state, a, pos):
    if a == lba.NORTH:
        return fruit_is_north(state, pos)
    if a == lba.SOUTH:
        return fruit_is_south(state, pos)
    if a == lba.EAST:
        return fruit_is_east(state, pos)
    if a == lba.WEST:
        return fruit_is_west(state, pos)
    return False


def extract_examples_from_demonstration_item(demonstration_item):
    """
    Convert a demonstrated (state, action) into positive and negative classification data.

    All actions not taken in the demonstration_item are considered negative.

    Parameters
    ----------
    demonstrations : (np.ndarray, int, (int, int))
        A state, action, agent position pair.

    Returns
    -------
    positive_examples : [(np.ndarray, int, (int, int))]
        A list with just the input state, action, position pair (for convenience).
    negative_examples : [(np.ndarray, int, (int, int))]
        A list with negative examples of state, action, positons paris
    """
    state, action, pos = demonstration_item

    positive_examples = [(state, action, pos)]
    negative_examples = []

    for a in [lba.NORTH, lba.SOUTH, lba.WEST, lba.EAST, lba.LOAD]:
        if a.value == action:
            continue

        if action == lba.LOAD.value:
            negative_examples.append((state, a.value, pos))

        if not is_fruit_direction_compatible_with_action(state, a, pos):
            negative_examples.append((state, a.value, pos))

    return positive_examples, negative_examples


def extract_examples_from_demonstration(demonstration):
    """
    Convert demonstrated (state, action)s into positive and negative classification data.

    Parameters
    ----------
    demonstrations : [(np.ndarray, (int, int))]
        State, action pairs

    Returns
    -------
    positive_examples : [(np.ndarray, (int, int))]
        A list with just the input state, action pairs (for convenience).
    negative_examples : [(np.ndarray, (int, int))]
        A list with negative examples of state, actions.
    """
    positive_examples = []
    negative_examples = []

    for demonstration_item in demonstration:
        demo_positive_examples, demo_negative_examples = extract_examples_from_demonstration_item(demonstration_item)
        positive_examples.extend(demo_positive_examples)
        negative_examples.extend(demo_negative_examples)

    return positive_examples, negative_examples


def apply_programs(programs, fn_input):
    """
    Worker function that applies a list of programs to a single given input.

    Parameters
    ----------
    programs : [ callable ]
    fn_input : Any

    Returns
    -------
    results : [ bool ]
        Program outputs in order.
    """
    x = []
    for program in programs:
        x_i = program(*fn_input)
        x.append(x_i)
    return x


# @manage_cache(cache_dir, ['.npz', '.pkl'])
def run_all_programs_on_single_demonstration(base_class_name, num_programs, demo_number, program_interval=1000):
    """
    Run all programs up to some iteration on one demonstration.

    Expensive in general because programs can be slow and numerous, so caching can be very helpful.

    Parallelization is designed to save time in the regime of many programs.

    Care is taken to avoid memory issues, which are a serious problem when num_programs exceeds 50,000.

    Returns classification dataset X, y.

    Parameters
    ----------
    base_class_name : str
    num_programs : int
    demo_number : int
    program_interval : int
        This interval splits up program batches for parallelization.

    Returns
    -------
    X : csr_matrix
        X.shape = (num_demo_items, num_programs)
    y : [ bool ]
        y.shape = (num_demo_items,)
    """

    print("Running all programs on {}, {}".format(base_class_name, demo_number))

    programs, _ = get_program_set(base_class_name, num_programs)

    demonstration = get_demonstrations(base_class_name, demo_numbers=(
        demo_number,), csv_file='collected-data/foraging-grid-8x8-2p-3f-v2/1654615226.csv')
    positive_examples, negative_examples = extract_examples_from_demonstration(demonstration)
    y = [1] * len(positive_examples) + [0] * len(negative_examples)

    num_data = len(y)
    num_programs = len(programs)

    X = lil_matrix((num_data, num_programs), dtype=bool)

    # This loop avoids memory issues
    for i in tqdm(range(0, num_programs, program_interval)):
        end = min(i+program_interval, num_programs)

        num_workers = multiprocessing.cpu_count()
        pool = multiprocessing.Pool(num_workers)

        fn = partial(apply_programs, programs[i:end])
        fn_inputs = positive_examples + negative_examples

        results = pool.map(fn, fn_inputs)
        pool.close()

        for X_idx, x in enumerate(results):
            X[X_idx, i:end] = x
    X = X.tocsr()
    print()
    return X, y


def run_all_programs_on_demonstrations(base_class_name, num_programs, demo_numbers):
    """
    See run_all_programs_on_single_demonstration.
    """
    X, y = None, None

    for demo_number in demo_numbers:
        demo_X, demo_y = run_all_programs_on_single_demonstration(base_class_name, num_programs, demo_number)

        if X is None:
            X = demo_X
            y = demo_y
        else:
            X = vstack([X, demo_X])
            y.extend(demo_y)

    y = np.array(y, dtype=np.uint8)

    return X, y


def learn_single_batch_decision_trees(y, num_dts, X_i):
    """
    Parameters
    ----------
    y : [ bool ]
    num_dts : int
    X_i : csr_matrix

    Returns
    -------
    clfs : [ DecisionTreeClassifier ]
    """
    clfs = []

    for seed in range(num_dts):
        clf = DecisionTreeClassifier(random_state=seed)
        clf.fit(X_i, y)
        # print(clf.score(X_i, y))
        clfs.append(clf)

    return clfs


def learn_plps(X, y, programs, program_prior_log_probs, num_dts=5, program_generation_step_size=10):
    """
    Parameters
    ----------
    X : csr_matrix
    y : [ bool ]
    programs : [ StateActionProgram ]
    program_prior_log_probs : [ float ]
    num_dts : int
    program_generation_step_size : int

    Returns
    -------
    plps : [ StateActionProgram ]
    plp_priors : [ float ]
        Log probabilities.
    """
    plps = []
    plp_priors = []

    num_programs = len(programs)

    for i in tqdm(range(0, num_programs, program_generation_step_size), desc="Learning plps with # programs"):
        for clf in learn_single_batch_decision_trees(y, num_dts, X[:, :i+1]):
            plp, plp_prior_log_prob = extract_plp_from_dt(clf, programs, program_prior_log_probs)
            plps.append(plp)
            plp_priors.append(plp_prior_log_prob)

    return plps, plp_priors


def compute_likelihood_single_plp(demonstrations, plp):
    """
    Parameters
    ----------
    demonstrations : [(np.ndarray, (int, int))]
        State, action pairs.
    plp : StateActionProgram

    Returns
    -------
    likelihood : float
        The log likelihood.
    """
    ll = 0.

    for obs, action, pos in demonstrations:
        # print(plp(obs, action, pos))
        if not plp(obs, action, pos):
            # instead of -np.inf
            return -np.inf

        size = 1

        for a in [lba.NONE, lba.NORTH, lba.SOUTH, lba.WEST, lba.EAST, lba.LOAD]:
            if a.value == action:
                continue
            if plp(obs, a, pos):
                size += 1

        ll += np.log(1. / size)

    return ll


def compute_likelihood_single_plp_reward(base_class_name, players, plp):
    ll = 0.0
    plp_policies = PLPPolicy([plp], [1.0])
    no_reward_count = 0
    for _ in range(100):
        if no_reward_count == 10:
            return -np.inf
        reward = run_foraging_policy(base_class_name, [plp_policies] * players, render=False, max_demo_length=20)
        if reward == 0.0:
            #no_reward_count += 1
            reward = 0.1
            # continue

        #reward += 0.00001
        ll += np.log(reward)
    return ll


def compute_likelihood_plps_reward(plps, base_class_name, players):
    """
    See compute_likelihood_single_plp.
    """
    num_workers = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(num_workers)

    fn = partial(compute_likelihood_single_plp_reward, base_class_name, players)
    likelihoods = pool.map(fn, plps)
    pool.close()

    return likelihoods


def compute_likelihood_plps(plps, demonstrations):
    """
    See compute_likelihood_single_plp.
    """
    num_workers = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(num_workers)

    fn = partial(compute_likelihood_single_plp, demonstrations)
    likelihoods = pool.map(fn, plps)
    pool.close()

    return likelihoods


def select_particles(particles, particle_log_probs, max_num_particles):
    """
    Parameters
    ----------
    particles : [ Any ]
    particle_log_probs : [ float ]
    max_num_particles : int

    Returns
    -------
    selected_particles : [ Any ]
    selected_particle_log_probs : [ float ]
    """
    sorted_log_probs, _, sorted_particles = (list(t)
                                             for t in zip(*sorted(zip(particle_log_probs, np.random.random(size=len(particles)), particles), reverse=True)))
    end = min(max_num_particles, len(sorted_particles))
    try:
        idx = sorted_log_probs.index(-np.inf)
        end = min(idx, end)
    except ValueError:
        pass
    return sorted_particles[:end], sorted_log_probs[:end]


# @manage_cache(cache_dir, '.pkl')
def train(base_class_name, demo_numbers, program_generation_step_size, num_programs, num_dts, max_num_particles):
    players = int(base_class_name[base_class_name.index('p')-1])
    programs, program_prior_log_probs = get_program_set(base_class_name, num_programs)

    X, y = run_all_programs_on_demonstrations(base_class_name, num_programs, demo_numbers)
    plps, plp_priors = learn_plps(X, y, programs, program_prior_log_probs, num_dts=num_dts,
                                  program_generation_step_size=program_generation_step_size)
    print(f"learned {len(plps)} plps")
    #demonstrations = get_demonstrations(base_class_name, demo_numbers=demo_numbers)
    likelihoods = compute_likelihood_plps_reward(plps, base_class_name, players)

    particles = []
    particle_log_probs = []

    for plp, prior, likelihood in zip(plps, plp_priors, likelihoods):
        #print(plp, prior, likelihood)
        print(prior, likelihood)
        particles.append(plp)
        particle_log_probs.append(likelihood)
    print("\nDone!")
    map_idx = np.argmax(particle_log_probs).squeeze()
    print("MAP program ({}):".format(particle_log_probs[map_idx]))
    # print(particles[map_idx])
    # print(plps[-1])
    top_particles, top_particle_log_probs = select_particles(particles, particle_log_probs, max_num_particles)
    if len(top_particle_log_probs) > 0:
        top_particle_log_probs = np.array(top_particle_log_probs) - logsumexp(top_particle_log_probs)
        top_particle_probs = np.exp(top_particle_log_probs)
        print("top_particle_probs:", top_particle_probs)
        policy = PLPPolicy(top_particles, top_particle_probs)
    else:
        print("no nontrivial particles found")
        policy = PLPPolicy([StateActionProgram("False")], [1.0])

    print('MODEL SELECTION:')
    best_plp_score = 0.0
    plps_score = []
    max_iter = 50
    iter_count = 0
    while max_iter > iter_count:
        rewards = []
        for d in range(100):
            rewards.append(run_foraging_policy(base_class_name, [policy]*players, render=False, max_demo_length=20))
        mean = np.array(rewards).mean()
        #print(f'avg. reward after run {d+1}: ' + str(mean))
        plps_score.append((policy, mean))
        if mean > best_plp_score:
            best_plp_score = mean
            print(f'new best avg. reward after run {d+1}: ' + str(best_plp_score))
        index_list = random.choices(range(len(plps)), k=max_num_particles)
        selected_plps = [plps[i] for i in index_list]
        selected_ll = [particle_log_probs[i] for i in index_list]
        policy = PLPPolicy(selected_plps, selected_ll)
        iter_count += 1

    sorted_plps = sorted(plps_score, key=lambda score: score[1], reverse=True)[:10]
    print('SCORES:')
    for score in sorted_plps:
        print(score[1])
    best_policies = [plp[0] for plp in sorted_plps[:10]]

    #p = random.choices(plps, k=max_num_particles)
    #p2 = random.choices(plps, k=max_num_particles)
    # return [PLPPolicy(p, len(p)*[-0.000000001]), PLPPolicy(p, len(p)*[-0.000000001])]
    return best_policies

# Test (given subset of environments)


def test(policy, base_class_name, test_env_nums=range(11, 20), max_num_steps=50,
         record_videos=True, video_format='mp4'):

    env_names = ['{}{}-v0'.format(base_class_name, i) for i in test_env_nums]
    envs = [gym.make(env_name) for env_name in env_names]
    accuracies = []
    for env in envs:
        video_out_path = '/tmp/lfd_{}.{}'.format(env.__class__.__name__, video_format)
        result = run_single_episode(env, policy, max_num_steps=max_num_steps,
                                    record_video=record_videos, video_out_path=video_out_path) > 0
        accuracies.append(result)

    return accuracies


if __name__ == "__main__":
    policy = train("TwoPileNim", range(11), 1, 31, 5, 25)
    test_results = test(policy, "TwoPileNim", range(11, 20), record_videos=True)
    print("Test results:", test_results)
