import numpy as np
import copy
from operator import itemgetter
from MCTS_alphazero import Node


def roll_out_function(state):
    action_probs = np.random.rand(len(state.allowedActions))
    return zip(state.allowedActions, action_probs)


def policy_value_function(state):
    action_probs = np.ones(len(state.allowedActions)) / len(state.allowedActions)
    return zip(state.allowedActions, action_probs)


class MCTS(object):
    def __init__(self, policy_value_fn, c_puct, n_simulation=10000):
        self.root = Node(None, 1.0)
        self.policy = policy_value_fn
        self.c_puct = c_puct
        self.n_simulation = n_simulation

    def _evaluate_rollout(self, state, limit=1000):
        player = state.player_turn
        winner = None
        for index in range(limit):
            winner = state.isEndGame
            if winner != 0:
                break
            action_probs = roll_out_function(state)
            max_action = max(action_probs, key=itemgetter(1))[0]
            state, _, _ = state.takeAction(max_action)
        else:
            print("WARNING: rollout reached move limit")
        if winner == 2:
            return 0
        else:
            return 1 if winner == player else -1

    def _simulation(self, state):
        node = self.root
        while True:
            if node.is_leaf():
                break
            action, node = node.select(self.c_puct)
            state, _, _ = state.takeAction(action)

        action_probs, _ = self.policy(state)
        winner = state.isEndGame
        if winner == 0:
            node.expand(action_probs)

        leaf_value = self._evaluate_rollout(state)
        node.update_recursive(-leaf_value)

    def get_move(self, state):
        for n in range(self.n_simulation):
            state_copy = copy.deepcopy(state)
            self._simulation(state_copy)
        return max(self.root.children.items(),
                   key=lambda act_node: act_node[1].n_visits)[0]

    def update_with_move(self, last_move):
        if last_move in self.root.children:
            self.root = self.root.children[last_move]
            self.root.parent = None
        else:
            self.root = Node(None, 1.0)

    def __str__(self):
        return "MCTS"


class MCTSPlayer(object):
    def __init__(self, c_puct=5, n_simulation=2000):
        self.mcts = MCTS(policy_value_function, c_puct, n_simulation)
        self.player = 1

    def set_player_ind(self, p):
        self.player = p

    def reset_player(self):
        self.mcts.update_with_move(-1)

    def get_action(self, state):
        sensible_moves = state.allowedActions
        if len(sensible_moves) > 0:
            move = self.mcts.get_move(state)
            self.mcts.update_with_move(-1)
            return move
        else:
            print("WARNING: the board full")

    def __str__(self):
        return f"MCTS {self.player}"



