import numpy as np
import copy
import config


def softmax(x):
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs


class Node(object):
    """
    MCTS树的节点，每个节点存储了当前的value Q, prior probability P 和 visit-count-adjusted prior score u
    """

    def __init__(self, parent, prior_p):
        self.parent = parent
        self.children = {}
        self.n_visits = 0
        self.Q = 0
        self.u = 0
        self.P = prior_p

    def expand(self, action_priors):
        """
        扩展叶子节点
        Args:
            action_priors: 由action和priors probability组成的tuple的list
        Returns:
        """
        for action, prob in action_priors:
            if action not in self.children:
                self.children[action] = Node(self, prob)

    def get_value(self, c_puct):
        """计算当前节点的Q+u"""
        # U = self.c_puct * ((1 - epsilon) * edge.stats['P'] + epsilon * nu[idx]) * np.sqrt(Nb) / (1 + edge.stats['N'])
        self.u = c_puct * self.P * np.sqrt(self.parent.n_visits) / (1 + self.n_visits)
        return self.u + self.Q

    def select(self, c_puct):
        """
        选择最大Q+u的action
        Args:
            c_puct: 常数
        Returns:
            A tuple of (action, next_node)
        """
        return max(self.children.items(), key=lambda act_node: act_node[1].get_value(c_puct))

    def update(self, leaf_value):
        self.n_visits += 1
        # 较奇怪
        self.Q += 1.0 * (leaf_value - self.Q) / self.n_visits

    def back_fill(self, leaf_value):
        if self.parent:
            self.parent.back_fill(-leaf_value)
        self.update(leaf_value)

    def is_leaf(self):
        return self.children == {}

    def is_root(self):
        return self.parent is None


class MCTS(object):
    def __init__(self, policy_value_function, c_puct, n_simulation):
        self.root = Node(None, 1.0)
        self.policy = policy_value_function
        self.c_puct = c_puct
        self.n_simulation = n_simulation

    def _simulation(self, state):
        node = self.root
        while not node.is_leaf():
            action, node = node.select(self.c_puct)
            state, _, _ = state.takeAction(action)

        action_probs, leaf_value = self.policy(state)

        winner = state.isEndGame

        if winner == 0:
            node.expand(action_probs)
        else:
            if winner == 2:  # tie
                leaf_value = 0.0
            else:
                leaf_value = (1.0 if winner == state.player_turn else -1.0)

        node.back_fill(-leaf_value)

    def get_move_probs(self, state, temp=1e-3):
        for n in range(self.n_simulation):
            state_copy = copy.deepcopy(state)
            self._simulation(state_copy)

        act_visits = [(act, node.n_visits) for act, node in self.root.children.items()]
        acts, visits = zip(*act_visits)
        act_probs = softmax(1.0 / temp * np.log(np.array(visits) + 1e-10))

        return acts, act_probs

    def update_with_move(self, last_move):
        if last_move in self.root.children:
            self.root = self.root.children[last_move]
            self.root.parent = None
        else:
            self.root = Node(None, 1.0)

    def __str__(self):
        return "MCTS"





