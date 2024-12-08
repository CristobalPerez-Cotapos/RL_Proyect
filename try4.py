import numpy as np
from connect_four_gymnasium import ConnectFourEnv
from connect_four_gymnasium.players import ConsolePlayer, BabyPlayer, ChildPlayer, TeenagerPlayer, AdultPlayer, AdultSmarterPlayer
from connect_four_gymnasium.tools import EloLeaderboard
from time import sleep


class Player:
    def __init__(self, name):
        self.name = name

    def play(self, observation):
        raise NotImplementedError("The 'play' method must be implemented in the child class")

    def getElo(self):
        return None
    
    def getName(self):
        return self.name

    def isDeterministic(self):
        raise NotImplementedError("The 'isDeterministic' method must be implemented in the child class")

class MinimaxPlayer(Player):
    def __init__(self, name="MinimaxPlayer", max_depth=4, heuristic=True):
        super().__init__(name)
        self.heuristic = heuristic
        self.max_depth = max_depth

    def play(self, obs):
        """
        Choose the best action for the current state.
        """
        if isinstance(obs, list):
            return [self._minimax_decision(o) for o in obs]
        else:
            return self._minimax_decision(obs)

    def _minimax_decision(self, observation):
        """
        Use the minimax algorithm with alpha-beta pruning to select the best move.
        """
        _, best_move = self._minimax(observation, self.max_depth, True, float('-inf'), float('inf'))
        return best_move

    def _minimax(self, board, depth, maximizing_player, alpha, beta):
        """
        Perform the minimax algorithm with alpha-beta pruning.
        """
        valid_moves = [c for c in range(7) if board[0, c] == 0]
        #print(valid_moves)
        if depth == 0 or not valid_moves:
            if self.heuristic:
                return self._evaluate_board(board), valid_moves[0] if valid_moves else None
            else:
                return self._simple_evaluate_board(board), valid_moves[0] if valid_moves else None


        if maximizing_player:
            max_eval = float('-inf')
            best_move = None
            for move in valid_moves:
                new_board, row, col = self._apply_move(board, move, 1)
                if self._check_win(new_board, 1, row, col):
                    return float('inf'), move
                eval_score, _ = self._minimax(new_board, depth - 1, False, alpha, beta)
                if eval_score > max_eval:
                    max_eval = eval_score
                    best_move = move
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break
            if best_move is None:
                best_move = valid_moves[0]
            return max_eval, best_move
        else:
            min_eval = float('inf')
            best_move = None
            for move in valid_moves:
                new_board, row, col = self._apply_move(board, move, -1)
                if self._check_win(new_board, -1, row, col):
                    return float('-inf'), move
                eval_score, _ = self._minimax(new_board, depth - 1, True, alpha, beta)
                if eval_score < min_eval:
                    min_eval = eval_score
                    best_move = move
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break
            if best_move is None:
                best_move = valid_moves[0]
            return min_eval, best_move

    def _evaluate_board(self, board):
        """
        Evaluate the board state using a multi-faceted heuristic.
        """
        score = 0

        # Positional weights (favor the center columns)
        center_array = [board[i, 3] for i in range(6)]
        center_count = center_array.count(1)
        score += center_count * 4  # Strong emphasis on center control

        # Evaluate all potential lines
        for r in range(6):
            for c in range(7):
                if board[r, c] == 0:
                    continue
                score += self._score_position(board, r, c, 1)  # Evaluate for player
                score -= self._score_position(board, r, c, -1)  # Evaluate for opponent

        # Additional heuristics for forks and blocks
        score += self._detect_forks(board, 1)  # Fork creation for the player
        score -= self._detect_forks(board, -1)  # Fork prevention for the opponent

        return score

    def _detect_forks(self, board, player):
        """
        Detect positions where the player can create a fork (multiple winning moves).
        """
        fork_score = 0
        for r in range(6):
            for c in range(7):
                if board[r, c] == 0:  # Check empty positions
                    simulated_board, row, col = self._apply_move(board, c, player)
                    winning_moves = 0
                    for move in range(7):
                        if simulated_board[0, move] == 0:
                            test_board, test_row, test_col = self._apply_move(simulated_board, move, player)
                            if self._check_win(test_board, player, test_row, test_col):
                                winning_moves += 1
                    if winning_moves > 1:  # Fork detected
                        fork_score += 50  # Prioritize forks heavily
        return fork_score

    def _score_position(self, board, row, col, player):
        """
        Score a specific position based on advanced patterns.
        """
        directions = [
            (1, 0),  # vertical
            (0, 1),  # horizontal
            (1, 1),  # diagonal /
            (1, -1)  # diagonal \
        ]
        score = 0
        for dr, dc in directions:
            line = []
            for step in range(-3, 4):
                r, c = row + step * dr, col + step * dc
                if 0 <= r < 6 and 0 <= c < 7:
                    line.append(board[r, c])
                else:
                    line.append(None)  # Out-of-bounds placeholder
            score += self._evaluate_line(line, player)
        return score

    def _evaluate_line(self, line, player):
        """
        Evaluate a line of 7 positions for advanced scoring.
        """
        score = 0
        for i in range(len(line) - 3):
            window = line[i:i + 4]
            if window.count(player) == 4:
                score += 1000  # Winning line
            elif window.count(player) == 3 and window.count(0) == 1:
                score += 15  # Threat: 3 in a row
            elif window.count(player) == 2 and window.count(0) == 2:
                score += 7  # Potential: 2 in a row
            elif window.count(-player) == 3 and window.count(0) == 1:
                score -= 20  # Block opponent's threat
        return score



    def _apply_move(self, board, move, player):
        """
        Apply a move to the board for the specified player.
        """
        new_board = board.copy()
        for i in range(5, -1, -1):
            if new_board[i, move] == 0:
                new_board[i, move] = player
                return new_board, i, move
        raise ValueError("Invalid move applied to a full column")

    def _check_win(self, board, player, row, col):
        """
        Check if the specified player has won given the last move.
        """
        directions = [
            (1, 0),  # horizontal
            (0, 1),  # vertical
            (1, 1),  # diagonal /
            (1, -1)  # diagonal \
        ]

        for dr, dc in directions:
            count = 0
            for step in range(-3, 4):
                r, c = row + step * dr, col + step * dc
                if 0 <= r < 6 and 0 <= c < 7 and board[r, c] == player:
                    count += 1
                    if count == 4:
                        return True
                else:
                    count = 0
        return False

    def getElo(self):
        """
        Estimated Elo rating for this player.
        """
        return 2000

    def isDeterministic(self):
        """
        Minimax player is deterministic.
        """
        return True


    def _simple_evaluate_board(self, board):
        """
        Evaluate the board state using a heuristic.
        """
        score = 0
        for move in range(7):
            if board[0, move] == 0:
                new_board, row, col = self._apply_move(board, move, 1)
                if self._check_win(new_board, 1, row, col):
                    score += 100
                elif self._check_win(new_board, -1, row, col):
                    score -= 100
        return score


# env = ConnectFourEnv(render_mode="human")
# opponent = MinimaxPlayer(max_depth=4, heuristic=False)
# env.change_opponent(opponent)
you = MinimaxPlayer(max_depth=2, heuristic=True)


# obs , _=  env.reset()
# for i in range(5000):
#     #sleep(0.5)
#     action = you.play(obs)
#     obs, rewards, dones, truncated,info = env.step(action)
#     env.render()
#     if(truncated or dones):
#         sleep(20)
#         obs , _=  env.reset()

elo = EloLeaderboard()
numero = elo.get_elo(you, parallel=True, num_matches=10)
print(numero)