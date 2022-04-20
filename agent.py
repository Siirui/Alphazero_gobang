# %matplotlib inline
import numpy as np
import random
import MCTS as mc
from game import GameState
import config
import loggers as lg
import time
import matplotlib.pyplot as plt
from IPython import display
import pylab as pl


class User(object):
    def __init__(self, name, state_size, action_size):
        self.name = name
        self.state_size = state_size
        self.action_size = action_size

    def act(self, state, tau):
        action = input('Enter your chosen action: ')
        pi = np.zeros(self.action_size)
        pi[action] = 1
        value = None
        NN_value = None

        return action, pi, value, NN_value


class Agent(object):
    def __init__(self, name, state_size, action_size, mcts_simulation, cpuct, model):
        self.root = None
        self.name = name
        self.state_size = state_size
        self.action_size = action_size
        self.cpuct = cpuct
        self.MCTS_simulations = mcts_simulation
        self.model = model

        self.mcts = None
        self.train_overall_loss = []
        self.train_value_loss = []
        self.train_policy_loss = []

        self.val_overall_loss = []
        self.val_value_loss = []
        self.val_policy_loss = []

    def buildMCTS(self, state):
        lg.logger_mcts.info(f"******BUILDING NEW MCTS TREE FOR AGENT {self.name} *******")
        self.root = mc.Node(state)
        self.mcts = mc.MCTS(self.root, self.cpuct)

    def changeRootMCTS(self, state):
        lg.logger_mcts.info(f"***** CHANGING ROOT OF MCTS TREE TO {state.id} FOR AGENT {self.name}")
        self.mcts.root = self.mcts.tree[state.id]

    def getPredictions(self, state):

        allowed_actions = state.allowedActions
        action_probs, leaf_value = self.model.policy_value_function(state)

        return leaf_value, action_probs, allowed_actions

    def evaluateLeaf(self, leaf, value, done, breadcrumbs):

        lg.logger_mcts.info("------EVALUATING LEAF------")

        if done == 0:
            value, probs, allowed_actions = self.getPredictions(leaf.state)
            lg.logger_mcts.info(f"PREDICTED VALUE FOR {leaf.state.player_turn}: {value}")

            for index, action in enumerate(allowed_actions):
                new_state, _, _ = leaf.state.takeAction(action)
                if new_state.id not in self.mcts.tree:
                    node = mc.Node(new_state)
                    self.mcts.addNode(node)
                    lg.logger_mcts.info(f"added node...{node.id}...p = {probs[index]}")
                else:
                    node = self.mcts.tree[new_state.id]
                    lg.logger_mcts.info(f"existing node...{node.id}...")

                new_edge = mc.Edge(leaf, node, probs[index], action)
                leaf.edges.append((action, new_edge))
        else:
            lg.logger_mcts.info(f"GAME VALUE FOR {leaf.player_turn}: {value}")

        return value, breadcrumbs

    def simulate(self):

        lg.logger_mcts.info(f"ROOT DONE...{self.mcts.root.state.id}")
        self.mcts.root.state.render(lg.logger_mcts)
        lg.logger_mcts.info(f"CURRENT PLAYER...{self.mcts.root.state.player_turn}")

        # Move to the leaf node
        leaf, value, done, breadcrumbs = self.mcts.moveToLeaf()
        leaf.state.render(lg.logger_mcts)

        # Evaluate the leaf node
        value, breadcrumbs = self.evaluateLeaf(leaf, value, done, breadcrumbs)

        # Backfill the value through the breadcrumbs
        self.mcts.backFill(leaf, value, breadcrumbs)

    def getAV(self, tau):
        edges = self.mcts.root.edges
        pi = np.zeros(self.action_size)
        values = np.zeros(self.action_size)

        for action, edge in edges:
            pi[action] = pow(edge.stats['N'], 1 / tau)
            values[action] = edge.stats['Q']
        pi = pi / (np.sum(pi) * 1.0)
        return pi, values

    def chooseAction(self, pi, values, tau):
        if tau == 0:
            actions = np.argwhere(pi == max(pi))
            action = random.choice(actions)[0]
        else:
            action_idx = np.random.multinomial(1, pi)
            action = np.where(action_idx == 1)[0][0]

        value = values[action]
        return action, value

    def act(self, state, tau):
        if self.mcts is None or state.id not in self.mcts.tree:
            self.buildMCTS(state)
        else:
            self.changeRootMCTS(state)

        # run the simulation
        for sim in range(self.MCTS_simulations):
            lg.logger_mcts.info('***************************')
            lg.logger_mcts.info('****** SIMULATION %d ******', sim + 1)
            lg.logger_mcts.info('***************************')
            self.simulate()

        # get action values
        pi, values = self.getAV(1)

        # pick the action
        action, value = self.chooseAction(pi, values, tau)

        next_state, _, _ = state.takeAction(action)
        NN_value = -self.getPredictions(next_state)[0]

        lg.logger_mcts.info(f"ACTION VALUES...{pi}")
        lg.logger_mcts.info(f"CHOOSE ACTION...{action}")
        lg.logger_mcts.info(f"MCTS PERCEIVED VALUE...{value}")
        lg.logger_mcts.info(f"NN PREDICTED VALUE...{NN_value}")

        return action, pi, value, NN_value




