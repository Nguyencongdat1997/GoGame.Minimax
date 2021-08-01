import numpy as np
import random

import sys
sys.path.append('../../')
from environment.go import GO, actions, black_stone, white_stone
from environment.game import Game
from players.base_player import BasePlayer
from players.q_players.base_learner import BaseLearner
from players.random_player.random_player import RandomPlayer
from players.greedy_player.greedy_player import GreedyPlayer
from players.minimax_players.minimax_player import MinimaxPlayer
from players.q_players.mc_every_visit_qlearner.qlearner import QLearner

class QTrainer:
    def __init__(self, learning_player:BaseLearner, learner_stone=black_stone, opponent_player=None, test_player=None):
        self.learning_player = learning_player
        self.opponent_player = opponent_player if opponent_player != None else GreedyPlayer()
        self.opponent_player = opponent_player if opponent_player != None else RandomPlayer()
        self.test_player = opponent_player if opponent_player != None else GreedyPlayer()
        self.learner_stone = learner_stone

        self.train_batch_test = 200
        self.batch_test = 5
        self.train_batch_save = 1000

    def train(self, iterations=1):
        self.learning_player.load_params()
        # print(len(self.learning_player.q_table), self.learning_player.q_table)
        print('Training with learner as {}....'.format('X' if self.learner_stone==black_stone else 'O'))
        count_played_games = 0
        count_black_wins = 0
        count_white_wins = 0
        for iteration in range(iterations):
            game = None
            if self.learner_stone == black_stone:
                game = Game(self.learning_player, self.opponent_player, verbose=False)
            else:
                game = Game(self.opponent_player, self.learning_player, verbose=False)

            winner, history = game.train()

            if winner == black_stone: count_black_wins += 1
            if winner == white_stone: count_white_wins += 1
            count_played_games += 1
            print('{} games have been trained. Current ratio X-tie-O: {}/{}/{}'
                  .format(count_played_games,
                          count_black_wins,
                          count_played_games - count_black_wins - count_white_wins,
                          count_white_wins)
                  )

            reward = 0
            if winner == self.learner_stone: reward = 1
            if winner == 3 - self.learner_stone: reward = -1
            episode_history = history['black'] if self.learner_stone == black_stone else history['white']
            self.learning_player.learn(episode_history=episode_history, reward=reward, board_size=game.board_size)

            if (iteration+1) % self.train_batch_test == 0:
                self.test(self.batch_test)
                # print('Number of trained states: ', len(self.learning_player.q_table))
            if (iteration+1) % self.train_batch_save == 0:
                self.learning_player.store_params()

        # print(len(self.learning_player.q_table), self.learning_player.q_table)
        # self.learning_player.store_params()

    def test(self, iterations=1):
        print('-- Testing....')
        count_played_games = 0
        count_black_wins = 0
        count_white_wins = 0
        for i in range(iterations):
            game = None
            if self.learner_stone == black_stone:
                game = Game(self.learning_player, self.test_player, verbose=False)
            else:
                game = Game(self.test_player, self.learning_player, verbose=False)
            winner = game.run()
            if winner == black_stone:
                count_black_wins += 1
            if winner == white_stone:
                count_white_wins += 1
            count_played_games += 1
            print('---- {} games have been tested for {} learner. Current black-tie-white ratio: {}/{}/{}'
                  .format(count_played_games,
                          'X' if self.learner_stone==black_stone else 'O',
                          count_black_wins,
                          count_played_games - count_black_wins - count_white_wins,
                          count_white_wins)
                  )
        print('-- Test Stopped....')
