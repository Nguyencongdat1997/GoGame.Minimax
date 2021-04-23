from .go import GO, white_stone, black_stone, actions
from .base_player import BasePlayer

class Game:
    def __init__(self, player1: BasePlayer, player2: BasePlayer, verbose=False):
        self.board_size = 5
        self.verbose = verbose
        self.go = GO(self.board_size)
        self.black_player = player1
        self.black_player.set_stone_type(black_stone)
        self.white_player = player2
        self.white_player.set_stone_type(white_stone)
        self.turn = black_stone
        self.game_ended = False
        self.winner = 0

    def run(self):
        if self.verbose:
            print('========Start new game========')
            while (not self.game_ended):
                print('Step ' + self.go.count_moved + ':')
                self.step()
            print('========Game Ended========')
        else:
            while (not self.game_ended):
                self.step
        return self.winner

    def step(self):
        if self.game_ended:
            return

        player = self.black_player if self.turn == black_stone else self.white_player
        action, x,y = player.play(self.go) # TODO: write this

        if action == actions['PLACE']:
            if not self.go.check_valid_placement(x, y, self.turn):
                print('Game end with invalid move.')
                winner = 3-self.turn
                print('The winner is {}'.format('X' if winner==black_stone else 'O'))
                self.game_ended = True
                self.winner = winner
                return
            else:
                self.go.move_forward(actions['PLACE'], (x,y), self.turn)

        if self.verbose:
            self.go.visualize_board()
            print()

        if self.go.check_game_end(self.turn, action):
            winner = self.go.get_winner()
            if self.verbose:
                print('Game end.')
                if winner == 0:
                    print('The game is a tie.')
                else:
                    print('The winner is {}'.format('X' if winner == black_stone else 'O'))
            self.game_ended = True
            self.winner = winner
            return
        else:
            self.turn = 3-self.turn
            return


