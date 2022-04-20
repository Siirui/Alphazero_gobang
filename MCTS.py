import numpy as np
import logging
import config
from utils import setup_logger
import loggers as lg


class Node(object):
    def __init__(self, state):
        self.state = state
        self.player_turn = state.player_turn
        self.id = state.id
        self.edges = []

    def isLeaf(self):
        if len(self.edges) > 0:
            return False
        else:
            return True


class Edge(object):
    def __init__(self, in_node, out_node, prior, action):
        self.id = in_node.id + '|' + out_node.id
        self.player_turn = in_node.player_turn
        self.action = action
        self.stat = {
            "N": 0,
            "W": 0,
            "Q": 0,
            "P": prior
        }


class MCTS(object):
    def __init__(self, root, c_puct):
        self.root = root
        self.tree = {}
        self.c_puct = c_puct
        self.addNode(root)

    def addNode(self, node):
        self.tree[node.id] = node

    def __len__(self):
        return len(self.tree)

    def moveToLeaf(self):

        lg.logger_mcts.info('------MOVING TO LEAF------')
        # a_t = argmax(Q(s_t, a) + u(s_t,a))
        # u(s,a) = c_puct * P(s,a) * sqrt(sum(N_r(s,b)) / (1 + N_r(s,a))

        breadcrumbs = []
        current_node = self.root
        done = 0
        value = 0

        while not current_node.isLeaf():
            maxQU = -99999

            if current_node == self.root:
                epsilon = config.EPSILON
                nu = np.random.dirichlet([config.ALPHA] * len(current_node.edges))
            else:
                epsilon = 0
                nu = [0] * len(current_node.edges)

            Nb = 0
            for action, edge in current_node.edges:
                Nb += edge.stats['N']

            simulation_action = None
            simulation_edge = None
            for idx, (action, edge) in enumerate(current_node.edges):
                U = self.c_puct * ((1 - epsilon) * edge.stats['P'] + epsilon * nu[idx]) * np.sqrt(Nb) / (1 + edge.stats['N'])
                Q = edge.stats['Q']
                lg.logger_mcts.info(f"action: {action} ({action % 7})...N = {edge.stats['N']}, "
                                    f"P = {np.round(edge.stats['P'], 6)}, nu = {np.round(nu[idx], 6)}, "
                                    f"adjP = {((1 - epsilon) * edge.stats['P'] + epsilon * nu[idx])}, "
                                    f"W = {np.round(edge.stats['W'], 6)}, Q = {np.round(Q, 6)}, "
                                    f"U = {np.round(U, 6)}, Q+U = {np.round(Q+U, 6)}")
                if Q + U > maxQU:
                    maxQU = Q + U
                    simulation_action = action
                    simulation_edge = edge

            lg.logger_mcts.info(f'action with highest Q+U...{simulation_action}')
            new_state, value, done = current_node.state.takeAction(simulation_action)
            current_node = simulation_edge.outNode
            breadcrumbs.append(simulation_edge)

        lg.logger_mcts.info(f"DONE...{done}")

        return current_node, value, done, breadcrumbs

    def backFill(self, leaf, value, breadcrumbs):

        lg.logger_mcts.info('------DOING BACKFILL------')

        current_player = leaf.state.player_turn

        for edge in breadcrumbs:
            player_turn = edge.player_turn
            if player_turn == current_player:
                direction = 1
            else:
                direction = -1

            edge.stats['N'] += 1
            edge.stats['W'] += value * direction
            edge.stats['Q'] = edge.stats['W'] / edge.stats['N']

            lg.logger_mcts.info(f"updating edge with value {value * direction} for player {player_turn}... "
                                f"N = {edge.stats['N']}, W = {edge.stats['W']}, Q = {edge.stats['Q']}")

            edge.out_node.state.render(lg.logger_mcts)
