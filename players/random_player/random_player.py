import random

import sys
sys.path.append('../../')
from environment.go import GO, actions
from players.base_player import BasePlayer

class RandomPlayer(BasePlayer):
    def __init__(self):
        super(RandomPlayer, self).__init__()
        self.type = 'Random'

    def play(self, go_game: GO):
        '''
            Randomly pick a placement in the game
        :param go_game: current game state
        :return:
            3 objects: action to move or pass, position x of placement, position y of placement
        '''
        possible_placements = go_game.get_possible_placements(self.stone_type)

        if not possible_placements:
            return actions['PASS'], -1, -1
        else:
            chosen_placement = random.choice(possible_placements)
            return actions['PLACE'], chosen_placement[0], chosen_placement[1]
