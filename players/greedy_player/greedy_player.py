import random

import sys
sys.path.append('../../')
from environment.go import GO, actions
from players.base_player import BasePlayer

NEGATIVE_INFI = -10000

class GreedyPlayer(BasePlayer):
    def __init__(self):
        super(GreedyPlayer, self).__init__()
        self.type = 'greedy'

    def play(self, go: GO):
        possible_placements = []
        if not go.check_game_end(self.stone_type, actions['PLACE']):
            possible_placements = go.get_possible_placements(self.stone_type)
        if not possible_placements:
            return actions["PASS"], -1, -1

        shuffled_move_indexes = [i for i in range(len(possible_placements))]
        if len(possible_placements) > 1:
            random.shuffle(shuffled_move_indexes)

        best_value = NEGATIVE_INFI
        best_move = None
        for move_index in shuffled_move_indexes:
            move = possible_placements[move_index]
            cur_go = go.copy_game()
            cur_go.move_forward(actions['PLACE'], move, self.stone_type)
            score = cur_go.get_reward(self.stone_type)
            if score >= best_value:
                best_value = score
                best_move = move
        return actions['PLACE'], best_move[0], best_move[1]
