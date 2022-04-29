from generalization_grid_games.envs import two_pile_nim as tpn
from generalization_grid_games.envs import checkmate_tactic as ct
from generalization_grid_games.envs import stop_the_fall as stf
from generalization_grid_games.envs import chase as ec
from generalization_grid_games.envs import reach_for_the_star as rfts
from lbforaging.foraging.environment import CellEntity as lbc, Action as lba

import generalization_grid_games


def get_object_types(base_class_name):
    if base_class_name == 'TwoPileNim':
        return ('tpn.EMPTY', 'tpn.TOKEN', 'None')
    if base_class_name == 'CheckmateTactic':
        return ('ct.EMPTY', 'ct.HIGHLIGHTED_WHITE_QUEEN', 'ct.BLACK_KING', 'ct.HIGHLIGHTED_WHITE_KING', 'ct.WHITE_KING', 'ct.WHITE_QUEEN', 'None')
    if base_class_name == 'StopTheFall':
        return ('stf.EMPTY', 'stf.FALLING', 'stf.RED', 'stf.STATIC', 'stf.ADVANCE', 'stf.DRAWN', 'None')
    if base_class_name == 'Chase':
        return ('ec.EMPTY', 'ec.TARGET', 'ec.AGENT', 'ec.WALL', 'ec.DRAWN', 'ec.LEFT_ARROW', 'ec.RIGHT_ARROW', 'ec.UP_ARROW', 'ec.DOWN_ARROW', 'None')
    if base_class_name == 'ReachForTheStar':
        return ('rfts.EMPTY', 'rfts.AGENT', 'rfts.STAR', 'rfts.DRAWN', 'rfts.LEFT_ARROW', 'rfts.RIGHT_ARROW', 'None')
    if base_class_name == 'ForagingEnv':
        # actions not used here as not in env obs:
        # 'lba.None', 'lba.NORTH', 'lba.SOUTH', 'lba.WEST', 'lba.EAST', 'lba.LOAD',
        # we also remove 'lbc.OUT_OF_BOUNDS', 'lbc.EMPTY.value', as our agent cannot move out of bounds?
        return ('lbc.FOOD.value', 'lbc.AGENT.value')

    raise Exception("Unknown class name", base_class_name)
