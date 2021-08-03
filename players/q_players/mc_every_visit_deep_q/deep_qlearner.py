import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Convolution2D
from tensorflow.keras.optimizers import Adam
import random
import os

import sys
sys.path.append('../../../')
from players.q_players.base_learner import BaseLearner
from environment.go import GO, actions


NEGATIVE_INFI = -1000


class DeepQNetwork(keras.Model):
    def __init__(self, n_actions):
        super(DeepQNetwork, self).__init__()

        fc1_dims = 128
        fc2_dims = 128
        self.dense1 = keras.layers.Dense(fc1_dims, activation='relu')
        self.dense2 = keras.layers.Dense(fc2_dims, activation='relu')
        self.Q = keras.layers.Dense(n_actions, activation=None)

    def call(self, state):
        x = self.dense1(state)
        x = self.dense2(x)
        Q = self.Q(x)
        return Q


class DeepQLearner(BaseLearner):
    def __init__(self, board_size, train_dir='./data', backup_strategy='random'):
        super(DeepQLearner, self).__init__()
        self.type = 'DQN_player'
        self.train_dir = train_dir
        self.learned_step_counter = 0

        n_actions = board_size ** 2+1
        gamma = 0.99
        epsilon_start = 1.0
        epsilon_dec = 1e-3
        epsilon_end = 0.01
        lr = 0.05

        self.action_space = [i for i in range(n_actions)]
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_dec = epsilon_dec
        self.epsilon_end = epsilon_end

        self.q = DeepQNetwork(n_actions)
        self.q.compile(optimizer=Adam(learning_rate=lr), loss='mean_squared_error')

    def play(self, go_game:GO):
        game_board = go_game.get_board().state
        game_state_encoded = self.encode_state(game_board, self.stone_type)

        if not(game_state_encoded in self.q_table):
            if self.backup_strategy == 'greedy':
                return self.play_greedy(go_game)
            return self.play_random(go_game)
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

    def learn(self, episode_history, reward, board_size):
        # reversed_history = episode_history
        # reversed_history.reverse()
        #
        # max_q_value_in_next_step = -1.0
        # value = reward
        # for step in reversed_history:
        #     self.learn_step += 1
        #     game_board, player_stone, action, x, y = step
        #     action_encoded = self.encode_action(action, (x, y), board_size)
        #     state_encoded = self.encode_state(game_board, player_stone)
        #     old_value = self._get_value(state_encoded, action_encoded)
        #     value *= self.gamma
        #     new_value = (1 - self.alpha) * old_value + self.alpha * value
        #     self._save_value(state_encoded, action_encoded, new_value, board_size=board_size)
        pass
        # print('Learned steps:', self.learn_step)

    def store_params(self):
        file_name = self.train_dir + '/dqn_' + str(self.learned_step_counter) + '/model'
        self.q.save_weights(file_name, save_format='tf')

    def load_params(self, learned_steps=100):
        file_name = self.train_dir + '/dqn_' + str(learned_steps) + '/model'
        if os.path.exists(file_name):
            self.q.load_weights(file_name)

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
        '''
        :param:
            board: 5x5 board of current game
            stone_type: current player stone
        :return:
            state: 1x25 vector, values in {-1,0,1}.
                value= 0 if no stone in that tile,
                value= 1 if the stone is the same with player stone
                value= -1 otherwise
        '''
        board_size = len(board)
        state = np.array(board)
        for x in range(board_size):
            for y in range(board_size):
                state[x][y] = -1 if state[x][y] == 3 - stone_type else state[x][y] == stone_type
        state = state.reshape((board_size**2))
        return state

