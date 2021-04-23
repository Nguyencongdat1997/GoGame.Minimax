from abc import ABC, abstractmethod

from .go import actions, GO

class BasePlayer(ABC):
    def __init__(self):
        super(BasePlayer, self).__init__()

    @abstractmethod
    def play(self, go_game: GO):
        return actions['PASS'], -1, -1

    def get_stone_type(self):
        return self.stone_type

    def set_stone_type(self, stone_type):
        self.stone_type = stone_type