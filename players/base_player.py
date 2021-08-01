from abc import ABC, abstractmethod
import sys
sys.path.append('../')
from environment.go import actions, GO

class BasePlayer(ABC):
    def __init__(self):
        super(BasePlayer, self).__init__()
        self.type = 'Not_defined'

    @abstractmethod
    def play(self, go_game: GO):
        '''
            Try to make a placement in the game
        :param go_game: current game state
        :return:
            3 objects: action to move or pass, position x of placement, position y of placement
        '''
        return actions['PASS'], -1, -1

    def get_stone_type(self):
        return self.stone_type

    def set_stone_type(self, stone_type):
        self.stone_type = stone_type