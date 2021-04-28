from abc import ABC, abstractmethod
import sys
sys.path.append('../../')
from players.base_player import BasePlayer
from environment.go import actions

class BaseLearner(BasePlayer):
    def __init__(self):
        super(BaseLearner, self).__init__()
        self.q_table = {}

    @abstractmethod
    def learn(self, episode_history):
        pass

    @abstractmethod
    def store_params(self):
        pass

    @abstractmethod
    def load_params(self):
        pass

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