from copy import deepcopy
import sys
sys.path.append('../')
from environment.constants import komi_value

""" Constants """
black_stone = 1
white_stone = 2
empty_tile = 0
actions = {'PASS': 0, 'PLACE':1}

""" Board """
class Board:

    def __init__(self, board_size = 5):
        self.size = board_size
        self.state = [[0 for x in range(board_size)] for y in range(board_size)] # Initialize 1 empty board

    def print_board(self):
        print('-' * self.size * 2)
        for i in range(self.size):
            for j in range(self.size):
                if self.state[i][j] == black_stone:
                    print('X', end=' ')
                elif self.state[i][j] == white_stone:
                    print('O', end=' ')
                else:
                    print(' ', end=' ')
            print()
        print('-' * self.size * 2)

    def compare_two_board(self, board_state1, board_state2):
        for i in range(self.size):
            for j in range(self.size):
                if board_state1[i][j] != board_state2[i][j]:
                    return False
        return True

    def compare_board(self, board2):
        return self.compare_two_board(self.state, board2.state)

    def have_liberties(self, i, j):
        '''
            Check if current position has any liberties (Having no liberty means it is a dead piece)
        :param i: position x
        :param j: position y
        :return:
            True if the position has any liberties
            False if the position has no liberties
        '''
        connected_allies = self.get_ally_by_dfs(i, j)
        for ally in connected_allies:
            neighbor_tiles_of_ally = self.get_neighbor_tiles(ally[0], ally[1])
            for tile in neighbor_tiles_of_ally:
                # If there is empty space around a piece, it has liberty
                if self.state[tile[0]][tile[1]] == empty_tile:
                    return True
        # If none of the pieces in a allied group has an empty space, it has no liberty
        return False

    def get_ally_by_dfs(self, i, j):
        '''
            Use DFS to find all the connected allies of current position
        :param i: position x
        :param j: position y
        :return:
            If there is no stone in this tile, return an empty array.
            Else return the array of positions of all its connected allies.
        '''
        stack = [(i, j)]  # stack for DFS serach
        connected_allies = []  # record allies positions during the search
        while stack:
            piece = stack.pop()
            connected_allies.append(piece)
            neighbor_allies = self.get_neighbor_allies(piece[0], piece[1])
            for ally in neighbor_allies:
                if ally not in stack and ally not in connected_allies:
                    stack.append(ally)
        return connected_allies

    def get_neighbor_allies(self, i, j):
        '''
            Check all neighbor tiles and return any tiles that are the allies of input position
        :param i: position x
        :param j: position y
        :return:
            Return all possible allies that are in the neighborhood of input position
        '''
        ally_negihbors = []
        neighbor_tiles = self.get_neighbor_tiles(i, j)
        for n in neighbor_tiles:
            # Add to allies list if having the same stone
            if self.state[n[0]][n[1]] == self.state[i][j]:
                ally_negihbors.append(n)
        return ally_negihbors

    def get_neighbor_tiles(self, i, j):
        '''
            Get 4 tiles that are the neighbor of input position
        :param i: position x
        :param j: position y
        :return:
            Return all possible neighbor tiles of the input position
        '''
        neighbor_tiles = []
        if i > 0: neighbor_tiles.append((i - 1, j))
        if i < len(self.state) - 1: neighbor_tiles.append((i + 1, j))
        if j > 0: neighbor_tiles.append((i, j - 1))
        if j < len(self.state) - 1: neighbor_tiles.append((i, j + 1))
        return neighbor_tiles

    def get_died_pieces(self, stone_type):
        died_pieces = []
        for i in range(self.size):
            for j in range(self.size):
                # Check if there is a piece at this position:
                if self.state[i][j] == stone_type:
                    # The piece dies if it has no liberty
                    if not self.have_liberties(i, j):
                        died_pieces.append((i, j))
        return died_pieces

    def update_state(self, new_sate):
        '''
            Update the board with new state
        '''
        # TODO: check the state size // assert len(new_sate) == self.size
        self.state = new_sate


""" Go Game """
class GO:
    def __init__(self, board_size = 5):
        # Unchanged values
        self.size = board_size
        self.max_move = board_size * board_size - 1  # The max movement of a Go game
        self.judge = Judge()

        # Changed values
        self._board = Board(board_size)
        self._previous_board = Board(board_size)
        self._died_pieces = []  # Intialize died pieces to be empty
        self.count_moved = 0  # Trace the number of moves

    def _get_opponent_type(self, piece_type):
        return 3 - piece_type

    def copy_game(self):
        return deepcopy(self)

    def get_possible_placements(self, piece_type):
        '''
            Return all possible placement of this piece_type in the current board
        '''
        moves = []
        for i in range(self.size):
            for j in range(self.size):
                if self.check_valid_placement(i, j, piece_type):
                    moves.append((i, j))
        return moves

    def visualize_board(self):
        self._board.print_board()

    def check_valid_placement(self, i, j, piece_type):
        '''
            Check whether a next placement is valid.
        :return: boolean indicating whether the placement is valid.
        '''
        # Check if the place is in the range of the board
        if not (i >= 0 and i < self.size):
            return False
        if not (j >= 0 and j < self.size):
            return False

        # Check if the place already has a piece
        if self._board.state[i][j] != 0:
            return False

        # Copy the board for testing
        next_go = self.copy_game() # the game after making this placement
        next_board = next_go._board

        # Check if the place has liberty
        next_board.state[i][j] = piece_type
        next_go._board.update_state(next_board.state)
        if next_go._board.have_liberties(i, j):
            return True
        # If not, remove the died pieces of opponent and check again
        next_go._remove_died_pieces(self._get_opponent_type(piece_type))
        if not next_go._board.have_liberties(i, j):
            return False

        # Check special case: repeat placement causing the repeat board state (KO rule)
        else:
            # TODO: recheck // if self.died_pieces and next_go.compare_2_board(self.previous_board, test_go.board):
            if next_go._board.compare_board(self._previous_board):
                return False
        return True

    def check_game_end(self, stone_type, move):
        '''
            Check if the game should end if making the next move.
        :param stone_type: black_stone or white_stone
        :param move: actions["MOVE"] or action["PASS"]
        :return:
            Boolean indicating whether the game should end.
        '''
        # Case 1: max move reached
        if self.count_moved >= self.max_move:
            # print("Max move reached by", piece_type)
            return True
        # Case 2: two players all pass the move.
        if move==actions['PASS'] and self._board.compare_board(self._previous_board)==True :
            print("Double pass by {}".format('X' if stone_type == black_stone else '0'))
            return True
        return False

    def move_forward(self, action, position, stone_type):
        if action != actions['PASS']:
            self._place_chess(position[0], position[1], stone_type)
            self._died_pieces = self._remove_died_pieces(self._get_opponent_type(stone_type))
        else:
            self._previous_board = deepcopy(self._board)
            self._died_pieces = []
        self.count_moved += 1 # TODO: consider if PASS should be counted as 1 move or not

    def _place_chess(self, i, j, stone_type):
        '''
            Trying to place a new stone to a position.
            If successfully placing, update the board state and return True
            Else return False.
        :param i: position x
        :param j: position y
        :param stone_type: white or black
        :return:
            If successfully placing  return True
            Else return False
        '''
        is_valid_placement = self.check_valid_placement(i, j, stone_type)
        if not is_valid_placement:
            return False
        board = self._board
        self._previous_board = deepcopy(board)
        board.state[i][j] = stone_type
        self._board.update_state(board.state)
        return True

    def _remove_died_pieces(self, piece_type):
        died_pieces = self._board.get_died_pieces(piece_type)
        if not died_pieces: return []
        self._remove_certain_pieces(died_pieces)
        return died_pieces

    def _remove_certain_pieces(self, positions):
        new_board_state = self._board.state
        for piece in positions:
            new_board_state[piece[0]][piece[1]] = 0
        self._board.update_state(new_board_state)

    def setup_game_state(self, previous_board:Board, board:Board, piece_type, move_count):
        '''
            update_game_state
            Setup a specific game state
        '''
        self._previous_board = previous_board
        self._board = board
        for i in range(self.size):
            for j in range(self.size):
                if self._previous_board.state[i][j] == piece_type and self._board.state[i][j] != piece_type:
                    self._died_pieces.append((i, j))
        self.count_moved = move_count

    # def set_board(self, piece_type, previous_board, board):
    #     for i in range(self.size):
    #         for j in range(self.size):
    #             if previous_board[i][j] == piece_type and board[i][j] != piece_type:
    #                 self.died_pieces.append((i, j))
    #
    #     # self.piece_type = piece_type
    #     self.previous_board = previous_board
    #     self.board = board

    def get_winner(self):
        return self.judge.judge_winner(self._board)

    def get_reward(self, player_type):
        return self._get_reward_by_win_lose(player_type)
        # TODO: setup other methods of reward

    def _get_reward_by_win_lose(self, player_type):
        if self.judge.judge_winner(self._board) == player_type:
            return 1
        elif self.judge.judge_winner(self._board) == 0:
            return 0
        return -1

    def _get_reward_by_captured_stones(self, player_type):
        return self.judge.compute_advanced_captured_stone_with_komi(player_type, self._board)



""" Judge """
class Judge:

    def __init__(self, komi=komi_value):
        self.komi = komi

    def judge_winner(self, board):
        '''
            Judge the winner of the game by number of pieces for each player.
        :param: None.
        :return: piece type of winner of the game (0 if it's a tie).
        '''
        black_score = self.compute_captured_stone(black_stone, board)
        white_score = self.compute_captured_stone(white_stone, board)
        if black_score > white_score + self.komi: return black_stone
        elif black_score < white_score + self.komi: return white_stone
        else: return 0

    def compute_captured_stone(self, stone_type, board: Board):
        '''
            Get score of a player by counting the number of stones.
        '''
        # TODO: consider using died pieces to calculate scores
        stone_count = 0
        for i in range(board.size):
            for j in range(board.size):
                if board.state[i][j] == stone_type:
                    stone_count += 1
        return stone_count

    def compute_advanced_captured_stone(self, player_type, board):
        '''
            Get the advance of score of current player in comparison with that of the opponent
        '''
        player_score = self.compute_captured_stone(player_type, board)
        opponent_score = self.compute_captured_stone(3 - player_type, board)
        return (player_score - opponent_score)

    def compute_advanced_captured_stone_with_komi(self, player_type, board):
        '''
            Get the advance of score of current player in comparison with that of the opponent, including komi value
        '''
        player_score = self.compute_captured_stone(player_type, board)
        opponent_score = self.compute_captured_stone(3 - player_type, board)
        if player_type == white_stone:
            player_score += self.komi
        else:
            opponent_score += self.komi
        return (player_score - opponent_score)

    def count_territory(self, stone_type, board: Board):
        count_territory = 0
        visited = []
        moves = [(0, 1), (1, 0), (0, -1), (-1, 0)]

        for tx in range(board.size):
            for ty in range(board.size):
                if (tx, ty) in visited:
                    continue
                visited.append((tx, ty))

                nexts = [(tx, ty)]
                territories = set()
                borders = set()
                while nexts:
                    cx, cy = nexts.pop(0)
                    if cx < 0 or cx >= board.size or cy < 0 or cy >= board.size:
                        continue
                    elif board.state[cx][cy] != '0':
                        continue
                    territories.append((cx, cy))

                    for my, mx in moves:
                        py, px = cy + my, cx + mx
                        if (py, px) in visited:
                            continue
                        elif board.state[px][py] == '0':
                            nexts.insert((py, px), 0)
                        else:
                            borders.add(board.state[px][py])
                if len(borders) == 1 and borders.pop() == stone_type:
                    count_territory += len(territories)
        return count_territory

    def count_board_liberty(self, stone_type, board: Board):
        '''
            Count the liberty of a stone type with simple method: if any neighbor tiles is the ally of current stone, the liberty is counted
        '''
        # TODO: when success, tranform this to calculate the minus of #of libterty of player and #of liberty of opponent at the save time
        liberty_count = 0
        for i in range(board.size):
            for j in range(board.size):
                if board.state[i][j] != empty_tile: # If this is not empty tile, it can not be counted as liberty
                    continue
                neighbors = board.get_neighbor_tiles(i, j)
                is_valid_liberty = 0
                for n in neighbors:
                    if board.state[n[0]][n[1]] == stone_type:
                        is_valid_liberty = 1
                        break
                liberty_count += is_valid_liberty
        return liberty_count

    def count_board_liberty_and_captured_stone(self, stone_type, board: Board):
        '''
            Count the score of a stone type, based on:
                - the Liberty: number of its liberties in the board
                - the Captured stone: number of position it has been captured
        '''
        score = 0
        for i in range(board.size):
            for j in range(board.size):
                if board.state[i][j] == stone_type: # If this position is captured by this stone_type
                    score += 1
                    continue
                if board.state[i][j] != empty_tile: # If this position is captured by the opponent
                    continue

                # If this position is not captured by anyone
                neighbors = board.get_neighbor_tiles(i, j)
                is_valid_liberty = 0
                for n in neighbors:
                    if self.board[n[0]][n[1]] == stone_type:
                        is_valid_liberty = 1
                        break
                score += is_valid_liberty
        return score

    def count_stones_on_edge(self, stone_type, board: Board):
        # Assume go.size >= 5
        count = 0
        for i in range(board.size):
            if board.state[i][0] == stone_type:
                count += 1
            if board.state[i][-1] == stone_type:
                count += 1
        for i in range(1, board.size - 1):
            if board.state[0][i] == stone_type:
                count += 1
            if board.state[-1][i] == stone_type:
                count += 1
        return count

    def euler_num(self, stone_type, board: Board):
        patterns_Q1 = [(0, 0, 0, 1), (0, 0, 1, 0), (0, 1, 0, 0), (1, 0, 0, 0)]
        patterns_Q3 = [(0, 1, 1, 1), (1, 0, 1, 1), (1, 1, 0, 1), (1, 1, 1, 0)]
        patterns_Qd = [(1, 0, 0, 1), (0, 1, 1, 0)]

        countQ1, countQ3, countQd = 0, 0, 0
        binary_board = [[1 if x == stone_type else 0 for x in rows] for rows in board.state]
        for y in range(0, board.size - 1):
            for x in range(0, board.size - 1):
                window = (
                binary_board[y][x], binary_board[y][x + 1], binary_board[y + 1][x], binary_board[y + 1][x + 1])
                if window in patterns_Q1:
                    countQ1 += 1
                if window in patterns_Q3:
                    countQ3 += 1
                if window in patterns_Qd:
                    countQd += 1
        euler_number = (countQ1 - countQ3 + 2 * countQd) / 4
        return euler_number
