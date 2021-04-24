from players.random_player.random_player import RandomPlayer
from players.minimax_players.minimax_player import MinimaxPlayer
from environment.game import  Game
from environment.go import black_stone, white_stone

if __name__ == "__main__":
    number_of_game = 20

    player1 = MinimaxPlayer()
    player2 = RandomPlayer()

    count_played_games = 0
    count_black_wins = 0
    count_white_wins = 0
    print('Running....')
    for i in range(number_of_game):
        game = Game(player1, player2, verbose=False)
        winner = game.run()
        if winner == black_stone:
            count_black_wins += 1
        if winner == white_stone:
            count_white_wins += 1
        count_played_games += 1
        print('{} games have been played. Current ratio: {}/{}/{}'
                .format(count_played_games,
                        count_black_wins,
                        count_played_games-count_black_wins-count_white_wins,
                        count_white_wins)
              )
    print('Stopped....')

