from MCTS_alphazero import MCTS
import numpy as np
import re

class MCTSPlayer(object):
    def __init__(self, policy_value_function, c_puct=5, n_simulation=2000, is_selfplay=0):
        self.mcts = MCTS(policy_value_function, c_puct, n_simulation)
        self.is_selfplay = is_selfplay
        self.player = None

    def set_player_ind(self, p):
        self.player = p

    def reset_player(self):
        self.mcts.update_with_move(-1)

    def get_action(self, state, temp=1e-3, return_prob=0):
        sensible_moves = state.allowedActions
        move_probs = np.zeros(state.grid_shape[0] * state.grid_shape[1])
        move = None
        if len(sensible_moves) > 0:
            acts, probs = self.mcts.get_move_probs(state, temp)
            move_probs[list(acts)] = probs
            if self.is_selfplay:
                move = np.random.choice(
                    acts,
                    p=0.75 * probs + 0.25 * np.random.dirichlet(0.3*np.ones(len(probs)))
                )
                self.mcts.update_with_move(move)
            else:
                move = np.random.choice(acts, p=probs)
                self.mcts.update_with_move(-1)

            if return_prob:
                return move, move_probs
            else:
                return move
        else:
            print("Warning: the board is full")

    def __str__(self):
        return f"MCTS {self.player}"


class HumanPlayer(object):
    def __init__(self, grid_shape):
        self.player = None
        self.grid_shape = grid_shape

    def set_player_ind(self, p):
        self.player = p

    def get_action(self, state):
        try:
            location = input("Your move: ")
            han = re.compile(",")
            move = re.split(han, location)
            move = int(move[0]) * self.grid_shape[1] + int(move[1])
        except Exception as e:
            move = -1
        if move == -1 or move not in state.allowedActions:
            print("invalid move")
            move = self.get_action(state)
        return move

    def __str__(self):
        return "Human {}".format(self.player)

