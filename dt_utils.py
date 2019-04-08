from policy import StateActionProgram

import numpy as np


def get_path_to_leaf(leaf, parents):
    reverse_path = []
    parent, parent_choice = parents[leaf]

    while True:
        reverse_path.append((parent, parent_choice))
        if parents[parent] is None:
            break
        parent, parent_choice = parents[parent]

    return reverse_path[::-1]

def get_conjunctive_program(path, node_to_features, features, feature_log_probs):
    program = '('
    log_p = 0.

    for i, (node_id, sign) in enumerate(path):
        feature_idx = node_to_features[node_id]
        precondition = features[feature_idx]
        feature_log_p = feature_log_probs[feature_idx]
        log_p += feature_log_p

        if sign == 'right':
            program = program + precondition
        else:
            assert sign == 'left'
            program = program + 'not (' + precondition + ')'

        if i < len(path) - 1:
            program = program + ' and '

    program = program + ')'

    return program, log_p

def get_disjunctive_program(conjunctive_programs):
    if len(conjunctive_programs) == 0:
        return 'False'

    program = ''

    for i, conjunctive_program in enumerate(conjunctive_programs):
        program = program +'(' + conjunctive_program + ')'
        if i < len(conjunctive_programs) - 1:
            program = program + ' or '

    return program

def extract_plp_from_dt(estimator, features, feature_log_probs):
    n_nodes = estimator.tree_.node_count
    children_left = estimator.tree_.children_left
    children_right = estimator.tree_.children_right
    node_to_features = estimator.tree_.feature
    threshold = estimator.tree_.threshold
    value = estimator.tree_.value.squeeze()

    stack = [0]
    parents = {0 : None}
    true_leaves = []

    while len(stack) > 0:
        node_id = stack.pop()

        if (children_left[node_id] != children_right[node_id]):
            assert 0 < threshold[node_id] < 1
            stack.append(children_left[node_id])
            parents[children_left[node_id]] = (node_id, 'left')
            stack.append(children_right[node_id])
            parents[children_right[node_id]] = (node_id, 'right')
        elif value[node_id][1] > value[node_id][0]:
            true_leaves.append(node_id)

    paths_to_true_leaves = [get_path_to_leaf(leaf, parents) for leaf in true_leaves]

    conjunctive_programs = []
    program_log_prob = 0.

    for path in paths_to_true_leaves:
        and_program, log_p = get_conjunctive_program(path, node_to_features, features, feature_log_probs)
        conjunctive_programs.append(and_program)
        program_log_prob += log_p

    disjunctive_program = get_disjunctive_program(conjunctive_programs)

    if not isinstance(disjunctive_program, StateActionProgram):
        disjunctive_program = StateActionProgram(disjunctive_program)

    return disjunctive_program, program_log_prob



