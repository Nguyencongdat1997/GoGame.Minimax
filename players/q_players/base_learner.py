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
    def learn(self, episode_history, reward, board_size):
        pass

    @abstractmethod
    def store_params(self):
        pass

    @abstractmethod
    def load_params(self):
        pass

    @abstractmethod
    def encode_action(self, action, move, board_size):
        pass

    @abstractmethod
    def decode_action(self, action, board_size):
        pass

    @abstractmethod
    def encode_state(self, board, stone_type):
        pass

    def decode_state(self, encoded_state):
        pass