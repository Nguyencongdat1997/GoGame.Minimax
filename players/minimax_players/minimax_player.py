import random

import sys
sys.path.append('../../')
from environment.go import GO, actions
from players.base_player import BasePlayer

NEGATIVE_INFI = -10000
POSITIVE_INFI = 10000

class MinimaxPlayer(BasePlayer):
    def __init__(self):
        super(MinimaxPlayer, self).__init__()
        self.type = 'Minimax'

        self.depth_limit = 3
        self.braching_factor_limit = 20

    def play(self, go_game: GO):
        temp_go = go_game.copy_game()
        _, best_move = self.minimax(temp_go, self.stone_type, self.depth_limit, True, NEGATIVE_INFI, POSITIVE_INFI)
        if (best_move == 'PASS'):
            return actions['PASS'], -1, -1
        else:
            return actions['PLACE'], best_move[0], best_move[1]

    def __evaluation_func(self, go_game:GO, stone_type):
        '''
            Evaluation func is used to score the resulted board when reaching limited depth of the search but the game has not ended.
            This needs to satisfy the characteristics of a Zero-Sum Game

            Here we use the reward function. That function will return 1 when the stone_type win, -1 when it is defeated and 0 when there is a tie
        :param go_game: current state of the game
        :return: the evaluation value.
        '''
        return go_game.get_reward(stone_type)

    def __reward_func(self, go_game: GO, stone_type):
        '''
            The reward function is used to score the result when the game ends.
            This needs to satisfy the characteristics of a Zero-Sum Game

            Here we use the reward function. That function will return 1 when the stone_type win, -1 when it is defeated and 0 when there is a tie
        :param go_game: current state of the game
        :return: the evaluation value.
        '''
        return go_game.get_reward(stone_type)

    def minimax(self,go:GO, stone_type, depth, is_maximizer, alpha, beta):
        if go.check_game_end(self.stone_type, actions['PLACE']):  # if limited num of moves reached
            reward = self.__reward_func(go, stone_type)
            return reward, "PASS"

        if depth < 1:  # if limited search depth reached
            heuristic_reward = self.__evaluation_func(go, stone_type)
            return heuristic_reward, "PASS"
            # TODO: need to return another action instead of PASS,
            #  ex: return reward, random.choice(possible_placements).
            #  But may be uncessary, because in this case, it is the value (not the action) that is focused

        possible_placements = go.get_possible_placements(stone_type)
        if not possible_placements:
            return self.__evaluation_func(go, stone_type), "PASS"

        shuffled_move_indexes = [i for i in range(len(possible_placements))] # Shuffle index make the move more randomly
        if len(possible_placements) > self.braching_factor_limit:
            shuffled_move_indexes = shuffled_move_indexes[:self.braching_factor_limit]
        random.shuffle(shuffled_move_indexes)

        if (is_maximizer):
            best_value = NEGATIVE_INFI
            # TODO: consider using PASS as boundary (as below), currently using PASS doest not work
            # if go.check_game_end(stone_type, actions["PASS"]):
            #     best_value = self.__reward_func(go, stone_type)
            best_move = "PASS"  # if no action in possible_placements or no one of them are better than PASS, take PASS

            for move_index in shuffled_move_indexes:  # Choose move through possible_placement until reach branching_factor_litmit
                move = possible_placements[move_index]

                cur_go = go.copy_game()
                cur_go.move_forward(actions['PLACE'], move, stone_type)
                next_piece_type = 3 - stone_type  # reverse turn
                value, _ = self.minimax(cur_go, next_piece_type, depth - 1, not is_maximizer, alpha, beta)
                value *= -1
                if value >= best_value:
                    best_value = value
                    best_move = move
                if best_value >= beta:
                    break
                alpha = max(alpha, best_value)
            return best_value, best_move
        else:
            best_value = POSITIVE_INFI
            # TODO: consider using PASS as boundary (as below), currently using PASS doest not work
            # if go.check_game_end(stone_type, actions["PASS"]):
            #     best_value = self.__reward_func(go, stone_type)
            best_move = "PASS"  # if no action in possible_placements or no one of them are better than PASS, take PASS

            for move_index in shuffled_move_indexes:
                move = possible_placements[move_index]

                cur_go = go.copy_game()
                cur_go.move_forward(actions['PLACE'], move, stone_type)
                next_piece_type = 3 - stone_type  # reverse turn
                value, _ = self.minimax(cur_go, next_piece_type, depth - 1, not is_maximizer, alpha, beta)
                value *= -1
                if value <= best_value:
                    best_value = value
                    best_move = move
                if best_value <= alpha:
                   break
                beta = min(beta, best_value)
            return best_value, best_move