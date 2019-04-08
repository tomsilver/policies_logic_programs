from dsl import *
from env_settings import *
from policy import StateActionProgram

from copy import deepcopy

import itertools
import heapq as hq
import pickle
import hashlib
import numpy as np



def find_symbol(program):
    for idx, elm in enumerate(program):
        if isinstance(elm, int):
            return elm, idx
        if isinstance(elm, list):
            rec_result = find_symbol(elm)
            if rec_result is not None:
                return rec_result[0], [idx, rec_result[1]]
    return None

def copy_program(program):
    return deepcopy(program)

def update_program(program, idx, new_symbol):
    if isinstance(idx, int):
        program[idx] = new_symbol
        return
    if len(idx) == 2:
        next_idx = idx[1]
    else:
        next_idx = idx[1:]
    update_program(program[idx[0]], next_idx, new_symbol)

def stringify(program):
    if isinstance(program, str):
        return program
    if isinstance(program, int):
        raise Exception("Should not stringify incomplete programs")
    s = ''
    for x in program:
        s = s + ' ' + stringify(x)
    return s.strip().lstrip()

def get_child_programs(program, grammar):
    symbol, idx = find_symbol(program)
    substitutions, production_probs = grammar[symbol]
    priorities = -np.log(production_probs)

    for substitution, prob, priority in zip(substitutions, production_probs, priorities):
        child_program = copy_program(program)
        update_program(child_program, idx, substitution)
        yield child_program, prob, priority

def program_is_complete(program):
    return find_symbol(program) == None

def generate_programs(grammar, start_symbol=0, num_iterations=100000000):
    queue = []
    counter = itertools.count()

    hq.heappush(queue, (0, 0, next(counter), [start_symbol]))

    for iteration in range(num_iterations):
        priority, production_neg_log_prob, _, program = hq.heappop(queue)

        for child_program, child_production_prob, child_priority in get_child_programs(program, grammar):
            if program_is_complete(child_program):
                yield StateActionProgram(stringify(child_program)), -production_neg_log_prob + np.log(child_production_prob)
            else:
                hq.heappush(queue, (priority + child_priority, production_neg_log_prob - np.log(child_production_prob), 
                                    next(counter), child_program))
