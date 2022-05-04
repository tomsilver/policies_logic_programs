import numpy as np
from lbforaging.foraging.environment import CellEntity as lbc, Action as lba


# Methods


def out_of_bounds(r, c, shape):
    return (r < 0 or c < 0 or r >= shape[0] or c >= shape[1])


def shifted(direction, local_program, action, obs, cell):
    if cell is None:
        new_cell = None
    else:
        new_cell = (cell[0] + direction[0], cell[1] + direction[1])
    return local_program(action, obs, new_cell)


def cell_is_value(value, action, obs, cell):
    if cell is None or out_of_bounds(cell[0], cell[1], obs[0].shape):
        return False

    # we need to check the value for which obs we need to use as we have one for food and agent
    # 0 is for agents, 1 for food

    if value == lbc.FOOD.value:
        focus = obs[1][cell[0], cell[1]]
        return focus > 0
    # elif value == lbc.AGENT.value:
    #    focus = obs[0][cell[0], cell[1]]
    #    return focus > 0
    else:
        return False


def at_cell_with_value(value, local_program, action, obs):
    if value == lbc.FOOD.value:
        matches = np.argwhere(obs[1] > 0)
    else:
        matches = []
    if len(matches) == 0:
        cell = None
    else:
        cell = matches[0]
    return local_program(action, obs, cell)

 # 'lba.None', 'lba.NORTH', 'lba.SOUTH', 'lba.WEST', 'lba.EAST', 'lba.LOAD',


def action_is_right(local_program, action, obs, pos):
    if action != lba.EAST.value:
        return False
    return local_program(action, obs, pos)


def action_is_up(local_program, action, obs, pos):
    if action != lba.NORTH.value:
        return False
    return local_program(action, obs, pos)


def action_is_down(local_program, action, obs, pos):
    if action != lba.SOUTH.value:
        return False
    return local_program(action, obs, pos)


def action_is_noop(local_program, action, obs, pos):
    if action != lba.LOAD.NONE:
        return False
    return local_program(action, obs, pos)


def action_is_left(local_program, action, obs, pos):
    if action != lba.WEST.value:
        return False
    return local_program(action, obs, pos)


def action_is_pickup(local_program, action, obs, pos):
    if action != lba.LOAD.value:
        return False
    return local_program(action, obs, pos)


def scanning(direction, true_condition, false_condition, action, obs, cell, max_timeout=50):
    if cell is None:
        return False

    for _ in range(max_timeout):
        cell = (cell[0] + direction[0], cell[1] + direction[1])

        if true_condition(action, obs, cell):
            return True

        if false_condition(action, obs, cell):
            return False

        # prevent infinite loops
        if out_of_bounds(cell[0], cell[1], obs[0].shape):
            return False

    return False


# Grammatical Prior
START, CONDITION, LOCAL_PROGRAM, DIRECTION, POSITIVE_NUM, NEGATIVE_NUM, VALUE = range(7)


def create_grammar(object_types):
    return {
        START: ([  # ['at_cell_with_value(', VALUE, ',', LOCAL_PROGRAM, ', a, s)'],
            # add methods for actions
            ['action_is_left(', LOCAL_PROGRAM, ', a, s, pos)'],
            ['action_is_right(', LOCAL_PROGRAM, ', a, s, pos)'],
            ['action_is_down(', LOCAL_PROGRAM, ', a, s, pos)'],
            ['action_is_up(', LOCAL_PROGRAM, ', a, s, pos)'],
            ['action_is_noop(', LOCAL_PROGRAM, ', a, s, pos)'],
            ['action_is_pickup(', LOCAL_PROGRAM, ', a, s, pos)']],
            6*[1/6]),
        LOCAL_PROGRAM: ([[CONDITION],
                         ['lambda a, o, pos : shifted(', DIRECTION, ',', CONDITION, ', a, o, pos)']],
                        [0.5, 0.5]),
        CONDITION: ([['lambda a, o, pos : cell_is_value(', VALUE, ', a, o, pos)'],
                     ['lambda a, o, pos: scanning(', DIRECTION, ',', LOCAL_PROGRAM, ',', LOCAL_PROGRAM, ', a, o, pos)']],
                    [0.5, 0.5]),
        DIRECTION: ([['(', POSITIVE_NUM, ', 0)'], ['(0,', POSITIVE_NUM, ')'],
                     ['(', NEGATIVE_NUM, ', 0)'], ['(0,', NEGATIVE_NUM, ')'],
                     ['(', POSITIVE_NUM, ',', POSITIVE_NUM, ')'], ['(', NEGATIVE_NUM, ',', POSITIVE_NUM, ')'],
                     ['(', POSITIVE_NUM, ',', NEGATIVE_NUM, ')'], ['(', NEGATIVE_NUM, ',', NEGATIVE_NUM, ')']],
                    [1./8] * 8),
        POSITIVE_NUM: ([['1'], [POSITIVE_NUM, '+1']],
                       [0.99, 0.01]),
        NEGATIVE_NUM: ([['-1'], [NEGATIVE_NUM, '-1']],
                       [0.99, 0.01]),
        VALUE: (object_types,
                [1./len(object_types) for _ in object_types])
    }
