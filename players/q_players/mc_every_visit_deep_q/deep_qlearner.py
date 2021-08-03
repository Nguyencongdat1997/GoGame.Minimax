import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import initializers
import tensorflow_addons as tfa
import random
import os

import sys

sys.path.append('../../../')
from players.q_players.base_learner import BaseLearner
from environment.go import GO, actions

NEGATIVE_INFI = -100000


class DeepQNetwork(keras.Model):
    def __init__(self, n_actions, board_size):
        super(DeepQNetwork, self).__init__()

        self.reshape = keras.layers.Reshape((board_size, board_size, 1), input_shape=(board_size, board_size))
        self.conv1 = keras.layers.Conv2D(64, 3)
        self.inorm1 = tfa.layers.InstanceNormalization(axis=3, center=True, scale=True,
                                                       beta_initializer="random_uniform",
                                                       gamma_initializer="random_uniform")
        self.conv2 = keras.layers.Conv2D(64, 2)
        self.conv3 = keras.layers.Conv2D(32, 2)
        self.flatten = keras.layers.Flatten()
        self.dense1 = keras.layers.Dense(128, activation='relu',
                                         kernel_initializer=initializers.RandomNormal(stddev=0.1))
        self.inorm1 = tfa.layers.InstanceNormalization(axis=3, center=True, scale=True,
                                                       beta_initializer="random_uniform",
                                                       gamma_initializer="random_uniform")
        self.dense2 = keras.layers.Dense(128, activation='relu',
                                         kernel_initializer=initializers.RandomNormal(stddev=0.1))
        self.Q = keras.layers.Dense(n_actions, activation=None,
                                         kernel_initializer=initializers.RandomNormal(stddev=0.1))

    def call(self, state):
        x = tf.cast(state, tf.float32)
        x = self.reshape(x)
        x = self.conv1(x)
        x = self.inorm1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        Q = self.Q(x)
        return Q


class DeepQLearner(BaseLearner):
    def __init__(self, board_size, gamma=0.99, beta=0.8, alpha=0.05, lr=0.005,
                 epsilon_start=1.0, epsilon_dec=1e-3, epsilon_end=0.01,
                 train_dir='./data', backup_strategy='random'):
        super(DeepQLearner, self).__init__()
        self.type = 'DQN_player'
        self.train_dir = train_dir
        self.backup_strategy = backup_strategy
        self.learned_step_counter = 0

        n_actions = board_size ** 2 + 1
        self.gamma = gamma  # coefficient for next_Q value
        self.alpha = alpha # coefficient for updating new Q
        self.beta = beta  # coefficient for accumulated reward
        self.epsilon = epsilon_start
        self.epsilon_dec = epsilon_dec
        self.epsilon_end = epsilon_end

        self.q = DeepQNetwork(n_actions, board_size=board_size)
        self.q.build(input_shape=(1, board_size, board_size))
        self.q.compile(optimizer=Adam(learning_rate=lr), loss='mean_squared_error')

    def play(self, go_game: GO):
        if np.random.random() < self.epsilon:
            if self.backup_strategy == 'greedy':
                return self.play_greedy(go_game)
            return self.play_random(go_game)
        else:
            game_board = go_game.get_board().state
            game_state_encoded = self.encode_state(game_board, self.stone_type)
            game_state_encoded = np.array([game_state_encoded])
            q_values = self.q(game_state_encoded)[0]

            min_q = min(q_values)
            max_q = max(q_values)
            if min_q >= max_q:
                # Situation when every q value = 0
                return self.play_random(go_game)

            possible_placements = go_game.get_possible_placements(self.stone_type)
            if not possible_placements:
                return actions['PASS'], -1, -1
            else:
                '''
                # TODO: here is a workaround, Try to use pass action later
                pass_action_encoded = self.encode_action(actions['PASS'], (-1, -1), go_game.size)                
                best_q = q_values[pass_action_encoded]
                best_move = (-1, -1)
                '''
                best_q = NEGATIVE_INFI
                best_move = (-1,-1)
                for move in possible_placements:
                    action_encoded = self.encode_action(actions['PLACE'], move, go_game.size)
                    q_value = q_values[action_encoded]
                    if q_value >= best_q:
                        best_q = q_value
                        best_move = move
                '''
                # TODO: here is a workaround, Try to use pass action later
                if best_move == (-1, -1):
                    return actions['PASS'], -1, -1
                else:
                    return actions['PLACE'], best_move[0], best_move[1]
                '''
                return actions['PLACE'], best_move[0], best_move[1]


    def play_random(self, go_game: GO):
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
        reversed_history = episode_history
        reversed_history.reverse()

        accumulated_reward = reward
        states_encoded = []
        actions_encoded = []
        rewards = []
        for step in range(len(reversed_history)):
            game_board, player_stone, action, x, y = reversed_history[step]
            state_encoded = self.encode_state(game_board, player_stone)
            states_encoded.append(state_encoded)
            action_encoded = self.encode_action(action, (x, y), board_size)
            actions_encoded.append(action_encoded)
            accumulated_reward *= self.beta
            rewards.append(accumulated_reward)

        states_encoded = np.array(states_encoded)
        q_pred = self.q(states_encoded)
        q_target = q_pred.numpy()
        max_next_actions = tf.math.argmax(q_pred, axis=1)
        for i in range(len(reversed_history)):
            if i > 0:
                max_q_next = rewards[i] + self.gamma * q_pred[i - 1, max_next_actions[i - 1]]
            else:
                max_q_next = rewards[i]
            q_target[i, actions_encoded[i]] = (1 - self.alpha) * q_target[i, actions_encoded[i]] \
                                              + self.alpha * max_q_next

        self.q.train_on_batch(states_encoded, q_target)

        self.learned_step_counter += len(episode_history)
        self.epsilon = max(self.epsilon - self.epsilon_dec, self.epsilon_end)
        print('Learned steps:', self.learned_step_counter, 'epsilon:', self.epsilon)

    def store_params(self):
        file_name = self.train_dir + '/dqn_' + str(self.learned_step_counter) + '/model'
        self.q.save_weights(file_name, save_format='tf')

    def load_params(self, learned_steps=100):
        file_name = self.train_dir + '/dqn_' + str(learned_steps) + '/model'
        if os.path.exists(file_name):
            self.q.load_weights(file_name)

    def encode_action(self, action, move, board_size):
        if action == actions['PASS']:
            return board_size ** 2
        return move[0] * board_size + move[1]

    def decode_action(self, action, board_size):
        '''
        :param action: a number, in range [0; board_size^2]
        :return: action_type, positon x, position y
        '''
        if action == board_size ** 2:
            return actions['PASS'], -1, -1
        else:
            y = action % board_size
            x = (int)((action - y) / board_size)
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
        return state
