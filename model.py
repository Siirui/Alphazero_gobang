import logging
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim

# import config


class Net(nn.Module):
    def __init__(self, grid_shape):
        super(Net, self).__init__()
        self.grid_shape = grid_shape
        # 卷积层
        self.conv1 = nn.Conv2d(4, 32, kernel_size=(3, 3), padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1)
        # action policy layers
        self.act_conv1 = nn.Conv2d(128, 4, kernel_size=(1, 1))
        self.act_fc1 = nn.Linear(4 * grid_shape[0] * grid_shape[1],
                                 grid_shape[0] * grid_shape[1])
        # state value layers
        self.val_conv1 = nn.Conv2d(128, 2, kernel_size=(1, 1))
        self.val_fc1 = nn.Linear(2 * grid_shape[0] * grid_shape[1],
                                 grid_shape[0] * grid_shape[1])
        self.val_fc2 = nn.Linear(grid_shape[0] * grid_shape[1], 1)

    def forward(self, state_input):
        # 卷积层
        x = F.relu(self.conv1(state_input))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        # action policy layers
        x_act = F.relu(self.act_conv1(x))
        x_act = x_act.view(-1, 4 * self.grid_shape[0] * self.grid_shape[1])
        x_act = F.log_softmax(self.act_fc1(x_act), dim=0)
        # state value layers
        x_val = F.relu(self.val_conv1(x))
        x_val = x_val.view(-1, 2 * self.grid_shape[0] * self.grid_shape[1])
        x_val = F.relu(self.val_fc1(x_val))
        x_val = F.relu(self.val_fc2(x_val))

        return x_act, x_val


class PolicyValueNet(object):
    def __init__(self, grid_shape, model_file=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.grid_shape = grid_shape
        self.l2_const = 1e-4
        self.policy_value_net = Net(grid_shape).to(self.device)
        self.optimizer = optim.Adam(self.policy_value_net.parameters(), weight_decay=self.l2_const)

        if model_file:
            net_params = torch.load(model_file)
            self.policy_value_net.load_state_dict(net_params)

    def policy_value(self, state_batch):
        """返回Action的概率和state value"""
        state_batch = torch.FloatTensor(state_batch, device=self.device)
        log_act_probs, value = self.policy_value_net(state_batch)
        act_probs = np.exp(log_act_probs.item())
        return act_probs, value.item()

    def policy_value_function(self, state):
        allowed_actions = state.allowedActions
        current_state = np.ascontiguousarray(state.get_current_state().reshape(
            -1, 4, self.grid_shape[0], self.grid_shape[1]))
        log_act_probs, value = self.policy_value_net(torch.from_numpy(current_state).to(self.device).float())
        act_probs = np.exp(log_act_probs.detach().numpy().flatten())
        act_probs = zip(allowed_actions, act_probs[allowed_actions])  # modified
        value = value.item()
        return act_probs, value

    def get_policy_params(self):
        net_params = self.policy_value_net.state_dict()
        return net_params

    def save_model(self, model_file):
        net_params = self.get_policy_params()
        torch.save(net_params, model_file)

    def train_step(self, state_batch, mcts_probs, winner_batch, lr):
        state_batch = torch.FloatTensor(state_batch).to(self.device)
        mcts_probs = torch.FloatTensor(mcts_probs).to(self.device)
        winner_batch = torch.FloatTensor(winner_batch).to(self.device)

        self.optimizer.zero_grad()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        log_act_probs, value = self.policy_value_net(state_batch)
        # loss = (z-v)^2 - pi^T * log(p) + c||theta||^2
        value_loss = F.mse_loss(value.view(-1), winner_batch)
        policy_loss = -torch.mean(torch.sum(mcts_probs * log_act_probs, 1))
        loss = value_loss + policy_loss
        loss.backward()
        self.optimizer.step()
        entropy = -torch.mean(torch.sum(torch.exp(log_act_probs) * log_act_probs, 1))
        return loss.item(), entropy.item()
