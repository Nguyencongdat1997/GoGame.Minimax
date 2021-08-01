import datetime

from players.random_player.random_player import RandomPlayer
from players.greedy_player.greedy_player import GreedyPlayer
from players.minimax_players.minimax_player import MinimaxPlayer
from players.minimax_players.negamax_player import NegamaxPlayer
from players.minimax_players.pvs_player import PVSPlayer
from players.q_players.mc_every_visit_qlearner.qlearner import QLearner
from environment.game import  Game
from environment.go import black_stone, white_stone

if __name__ == "__main__":
    number_of_game = 50

    player1 = QLearner()
    player1.load_params()
    player2 = RandomPlayer()

    # player1 = MinimaxPlayer()
    # player2 = RandomPlayer()

    count_played_games = 0
    count_black_wins = 0
    count_white_wins = 0

    star_time = datetime.datetime.now()
    print('Running....')
    for i in range(number_of_game):
        game = Game(player1, player2, verbose=False)
        winner = game.run()
        if winner == black_stone:
            count_black_wins += 1
        if winner == white_stone:
            count_white_wins += 1
        count_played_games += 1
        print('{} games have been played between {} as X and {} as O. Current ratio X-tie-O: {}/{}/{}'
                .format(count_played_games,
                        player1.type,
                        player2.type,
                        count_black_wins,
                        count_played_games-count_black_wins-count_white_wins,
                        count_white_wins)
              )
    print('Stopped....')
    end_time = datetime.datetime.now()
    print('After {} seconds'.format(end_time-star_time))
