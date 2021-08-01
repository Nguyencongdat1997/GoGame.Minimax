import csv
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
import random
import os

import sys
sys.path.append('../../../')
from players.q_players.base_learner import BaseLearner
from environment.go import GO, actions


NEGATIVE_INFI = -10000


class QLearner(BaseLearner):
    def __init__(self, param_file='./data/qtable.csv'):
        super(QLearner, self).__init__()
        self.stored_file = param_file
        self.alpha = .3
        self.gamma = .4
        self.q_table = {}

        self.learn_step = 0

    def play(self, go_game:GO):
        game_board = go_game.get_board().state
        game_state_encoded = self.encode_state(game_board, self.stone_type)

        if not(game_state_encoded in self.q_table):
            return self.play_greedy(go_game)
        else:
            values = self.q_table[game_state_encoded]
            possible_placements = go_game.get_possible_placements(self.stone_type)
            if not possible_placements:
                return actions['PASS'], -1, -1
            else:
                best_value = NEGATIVE_INFI
                best_move = actions['PASS']
                for move in possible_placements:
                    action_encoded = self.encode_action(actions['PLACE'], move, go_game.size)
                    value = values[action_encoded]
                    if value >= best_value:
                        best_value = value
                        best_move = move
                return actions['PLACE'], best_move[0], best_move[1]

    def play_random(self, go_game:GO):
        possible_placements = go_game.get_possible_placements(self.stone_type)
        if not possible_placements:
            return actions['PASS'], -1, -1
        else:
            chosen_placement = random.choice(possible_placements)
            return actions['PLACE'], chosen_placement[0], chosen_placement[1]

    def play_greedy(self, go_game: GO):
        possible_placements = []
        if not go_game.check_game_end(self.stone_type, actions['PLACE']):
            possible_placements = go_game.get_possible_placements(self.stone_type)
        if not possible_placements:
            return actions["PASS"], -1, -1

        shuffled_move_indexes = [i for i in range(len(possible_placements))]
        if len(possible_placements) > 1:
            random.shuffle(shuffled_move_indexes)

        best_value = NEGATIVE_INFI
        best_move = None
        for move_index in shuffled_move_indexes:
            move = possible_placements[move_index]
            cur_go = go_game.copy_game()
            cur_go.move_forward(actions['PLACE'], move, self.stone_type)
            score = cur_go.get_reward(self.stone_type)
            if score >= best_value:
                best_value = score
                best_move = move
        return actions['PLACE'], best_move[0], best_move[1]

    def _save_value(self, game_state_encoded, action_encoded, value, board_size):
        if game_state_encoded in self.q_table:
            self.q_table[game_state_encoded][action_encoded] = value
        else:
            self.q_table[game_state_encoded] = [0 for i in range(board_size**2+1)]
            self.q_table[game_state_encoded][action_encoded] = value

    def _get_value(self, game_state_encoded, action_encoded):
        if game_state_encoded in self.q_table:
            return self.q_table[game_state_encoded][action_encoded]
        return 0

    def learn(self, episode_history, reward, board_size):
        reversed_history = episode_history
        reversed_history.reverse()
        max_q_value_in_next_step = -1.0
        value = reward
        for step in reversed_history:
            self.learn_step += 1
            game_board, player_stone, action, x, y = step
            action_encoded = self.encode_action(action, (x, y), board_size)
            state_encoded = self.encode_state(game_board, player_stone)

            old_value = self._get_value(state_encoded, action_encoded)

            # # TODO: update from this V value to Q value
            # if max_q_value_in_next_step < 0: # if in last step, update directly
            #     new_value = reward
            # else:
            #     new_value = (1 - self.alpha) * old_value + self.alpha * self.gamma * max_q_value_in_next_step
            # #max_q_value_in_next_step = ...

            value *= self.gamma
            new_value = (1 - self.alpha) * old_value + self.alpha * value

            self._save_value(state_encoded, action_encoded, new_value, board_size=board_size)
        #print('Learned steps:', self.learn_step)

    def store_params(self):
        # with open(self.stored_file, 'w') as f:
        #     for key in self.q_table.keys():
        #         f.write("%s,%s\n" % (key, self.q_table[key]))
        df = pd.DataFrame({key: pd.Series(value) for key, value in self.q_table.items()})
        if not os.path.exists(self.stored_file):
            os.mkdir('./data')
        df.to_csv(self.stored_file, encoding='utf-8', index=False)

    def load_params(self, param_file=None):
        if param_file == None:
            param_file = self.stored_file
        # with open(param_file, mode='r') as infile:
        #     reader = csv.reader(infile)
        #     self.q_table = {rows[0]: rows[1] for rows in reader}3
        if os.path.exists(self.stored_file):
            df = pd.read_csv(param_file)
            df = df.apply(lambda x: list(x)).to_dict()
            self.q_table = df
        else:
            self.q_table = {}

    def encode_action(self, action, move, board_size):
        if action == actions['PASS']:
            return board_size**2
        return move[0]*board_size + move[1]

    def decode_action(self, action, board_size):
        '''

        :param action: a number, in range [0; board_size^2]
        :return: action_type, positon x, position y
        '''
        if action == board_size ** 2:
            return actions['PASS'], -1, -1
        else:
            y = action % board_size
            x = (int) ((action-y)/board_size)
            return action['PLACE'], x, y

    def encode_state(self, board, stone_type):
        txt_state = ''.join([str(y) for x in board for y in x])
        txt_state += str(stone_type)
        return txt_state
