import numpy as np
from lbforaging.foraging.environment import CellEntity as lbc, Action as lba

# Methods


def closest_node(node, nodes, mask):
    nodes = np.asarray(nodes)
    dist_2 = np.sum((nodes - node)**2, axis=1)
    dist = np.ma.MaskedArray(dist_2, mask)
    return np.ma.argmin(dist)


def get_new_agent_pos(pos, action):
    if action == lba.NORTH.value:
        return (pos[0] - 1, pos[1])
    elif action == lba.SOUTH.value:
        return (pos[0] + 1, pos[1])
    elif action == lba.WEST.value:
        return (pos[0], pos[1] - 1)
    elif action == lba.EAST.value:
        return (pos[0], pos[1] + 1)
    else:
        return pos


def find_nearest_pickable_food(obs, pos):
    agent, food = obs
    agent_lvl = agent[pos[0], pos[1]]
    food_pos = np.argwhere(food > 0)
    food_lvls = [food[p[0], p[1]] for p in food_pos]
    can_not_pickup = [fl > agent_lvl for fl in food_lvls]
    if all(can_not_pickup):
        can_not_pickup = [False for _ in food_lvls]
        nearest_food = food_pos[closest_node(pos, food_pos, can_not_pickup)]
    else:
        nearest_food = food_pos[closest_node(pos, food_pos, can_not_pickup)]
    return nearest_food


def action_is_executable(action, obs, pos):
    new_pos = get_new_agent_pos(pos, action)
    agent, food = obs
    try:
        return agent[new_pos] == 0 and food[new_pos] == 0
    except IndexError:
        return False


def shifted(direction, local_program, action, obs, cell):
    if cell is None:
        new_cell = None
    else:
        new_cell = (cell[0] + direction[0], cell[1] + direction[1])

    if out_of_bounds(new_cell[0], new_cell[1], obs[0].shape):
        return False
    return local_program(action, obs, new_cell)


def cell_is_value(value, action, obs, cell):
    if cell is None or out_of_bounds(cell[0], cell[1], obs[0].shape):
        return False

    # we need to check the value for which obs we need to use as we have one for food and agent
    # 0 is for agents, 1 for food

    if value == lbc.FOOD.value:
        focus = obs[1][cell[0], cell[1]]
        return focus > 0
    elif value == lbc.AGENT.value:
        focus = obs[0][cell[0], cell[1]]
        return focus > 0
    elif value == lbc.EMPTY.value:
        focus_agent = obs[0][cell[0], cell[1]]
        focus_food = obs[1][cell[0], cell[1]]
        return focus_agent + focus_food == 0
    else:
        return False


def out_of_bounds(r, c, shape):
    return (r < 0 or c < 0 or r >= shape[0] or c >= shape[1])


def fruit_is_pickable(obs, pos):
    nearest_food = find_nearest_pickable_food(obs, pos)
    return abs(nearest_food[1] - pos[1]) + abs(nearest_food[0] - pos[0]) == 1


def fruit_is_east(obs, pos):
    nearest_food = find_nearest_pickable_food(obs, pos)
    return nearest_food[1] - pos[1] > 0 and abs(nearest_food[0] - pos[0]) > 0


def fruit_is_south(obs, pos):
    nearest_food = find_nearest_pickable_food(obs, pos)
    return nearest_food[0] - pos[0] > 0 and abs(nearest_food[1] - pos[1]) > 0


def fruit_is_west(obs, pos):
    nearest_food = find_nearest_pickable_food(obs, pos)
    return pos[1] - nearest_food[1] > 0 and abs(nearest_food[0] - pos[0]) > 0


def fruit_is_north(obs, pos):
    nearest_food = find_nearest_pickable_food(obs, pos)
    return pos[0] - nearest_food[0] > 0 and abs(nearest_food[1] - pos[1]) > 0


def action_is_east(local_program, action, obs, pos):
    if action != lba.EAST.value:
        return False
    return local_program(action, obs, pos)


def action_is_north(local_program, action, obs, pos):
    if action != lba.NORTH.value:
        return False
    return local_program(action, obs, pos)


def action_is_south(local_program, action, obs, pos):
    if action != lba.SOUTH.value:
        return False
    return local_program(action, obs, pos)


def action_is_noop(local_program, action, obs, pos):
    if action != lba.LOAD.NONE:
        return False
    return local_program(action, obs, pos)


def action_is_west(local_program, action, obs, pos):
    if action != lba.WEST.value:
        return False
    return local_program(action, obs, pos)


def action_is_load(local_program, action, obs, pos):
    if action != lba.LOAD.value:
        return False
    return local_program(action, obs, pos)


# Grammatical Prior
START, CONDITION, LOCAL_PROGRAM, DIRECTION, POSITIVE_NUM, NEGATIVE_NUM, VALUE = range(7)


def create_grammar(object_types):
    return {
        START: ([
            # add methods for actions
            ['action_is_west(', LOCAL_PROGRAM, ', a, s, pos)'],
            ['action_is_east(', LOCAL_PROGRAM, ', a, s, pos)'],
            ['action_is_south(', LOCAL_PROGRAM, ', a, s, pos)'],
            ['action_is_north(', LOCAL_PROGRAM, ', a, s, pos)'],
            ['action_is_noop(', LOCAL_PROGRAM, ', a, s, pos)'],
            ['action_is_load(', LOCAL_PROGRAM, ', a, s, pos)']],
            6*[1/6]),
        LOCAL_PROGRAM: ([[CONDITION],
                         ['lambda a, o, pos : shifted(', DIRECTION, ',', CONDITION, ', a, o, pos)']],
                        [0.5, 0.5]),
        CONDITION: ([['lambda a, o, pos : cell_is_value(', VALUE, ', a, o, pos)'],
                     ['lambda a, o, pos : fruit_is_east(o, pos)'],
                     ['lambda a, o, pos : fruit_is_south(o, pos)'],
                     ['lambda a, o, pos : fruit_is_west(o, pos)'],
                     ['lambda a, o, pos : fruit_is_north(o, pos)'],
                     ['lambda a, o, pos : fruit_is_pickable(o, pos)'],
                     ['lambda a, o, pos : action_is_executable(a, o, pos)']],
                    7*[1/7]),
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
