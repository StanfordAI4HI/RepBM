import torch.nn as nn
import torch
import copy
from torch.autograd import Variable
import torch.nn.functional as F
import math


class QNet(nn.Module):
    def __init__(self, state_dim, hidden_dims, action_size):
        super(QNet, self).__init__()
        mlp_layers = []
        prev_hidden_size = state_dim
        for next_hidden_size in hidden_dims:
            mlp_layers.extend([
                nn.Linear(prev_hidden_size, next_hidden_size),
                nn.Tanh(),
            ])
            prev_hidden_size = next_hidden_size
        mlp_layers.append(
            nn.Linear(prev_hidden_size, action_size)
        )
        self.model = nn.Sequential(*mlp_layers)

    def forward(self, state):
        return self.model(state)


class QtNet(nn.Module):
    def __init__(self, state_dim, hidden_dims, action_size):
        super(QtNet, self).__init__()
        self.action_size = action_size
        self.state_dim = state_dim
        mlp_layers = []
        prev_hidden_size = state_dim
        for next_hidden_size in hidden_dims:
            mlp_layers.extend([
                nn.Linear(prev_hidden_size, next_hidden_size),
                nn.Tanh(),
            ])
            prev_hidden_size = next_hidden_size
        mlp_layers.append(
            nn.Linear(prev_hidden_size, action_size)
        )
        self.model = nn.Sequential(*mlp_layers)
        self.time_weights = nn.Linear(1, 1)

    def forward(self, state, time):
        #f = torch.cat((state,time),1)
        #return self.model(f)
        return self.model(state)+self.time_weights(time).repeat(1,self.action_size)


class MDPnet(nn.Module):
    def __init__(self, config):
        super(MDPnet, self).__init__()
        self.config = config
        # representation
        mlp_layers = []
        prev_hidden_size = config.state_dim
        for next_hidden_size in config.rep_hidden_dims:
            mlp_layers.extend([
                nn.Linear(prev_hidden_size, next_hidden_size),
                nn.Tanh(),
            ])
            prev_hidden_size = next_hidden_size
        self.representation = nn.Sequential(*mlp_layers)
        self.rep_dim = prev_hidden_size

        # Transition
        mlp_layers = []
        for next_hidden_size in config.transition_hidden_dims:
            mlp_layers.extend([
                nn.Linear(prev_hidden_size, next_hidden_size),
                nn.Tanh(),
            ])
            prev_hidden_size = next_hidden_size
        mlp_layers.append(
            nn.Linear(prev_hidden_size,config.action_size*config.state_dim)
        )
        self.transition = nn.Sequential(*mlp_layers)

        #Reward
        mlp_layers = []
        prev_hidden_size = self.rep_dim
        for next_hidden_size in config.reward_hidden_dims:
            mlp_layers.extend([
                nn.Linear(prev_hidden_size, next_hidden_size),
                nn.Tanh(),
            ])
            prev_hidden_size = next_hidden_size
        mlp_layers.append(
            nn.Linear(prev_hidden_size, config.action_size)
        )
        self.reward = nn.Sequential(*mlp_layers)

    def forward(self,state):
        rep = self.representation(state)
        next_state_diff = self.transition(rep).view(-1,self.config.action_size,self.config.state_dim)
        reward = self.reward(rep).view(-1,self.config.action_size)
        #soft_done = self.terminal(state)
        return next_state_diff, reward, rep

    # oracle for cartpole, we should not use
    def get_isdone(self,state):
        x = state[0,0]
        theta = state[0,2]
        x_threshold = 2.4
        theta_threshold_radians = 12*2*math.pi/360
        done = x < -x_threshold \
               or x > x_threshold \
               or theta < -theta_threshold_radians \
               or theta > theta_threshold_radians
        done = bool(done)
        return done
    # oracle for cartpole, we should not use
    def get_reward(self,state):
        return self.config.oracle_reward


class TerminalClassifier(nn.Module):
    def __init__(self, config):
        super(TerminalClassifier, self).__init__()
        self.config = config

        mlp_layers = []
        prev_hidden_size = self.config.state_dim #self.config.rep_hidden_dims[-1]
        for next_hidden_size in config.terminal_hidden_dims:
            mlp_layers.extend([
                nn.Linear(prev_hidden_size, next_hidden_size),
                nn.Tanh(),
            ])
            prev_hidden_size = next_hidden_size
        mlp_layers.extend([
            nn.Linear(prev_hidden_size, 1),
        ])
        self.terminal = nn.Sequential(*mlp_layers)

    def forward(self, x):
        return self.terminal(x)


class TabularMDPnet(nn.Module):
    def __init__(self, config, feature_table):
        # Feature_table is a tensor of size (state_size x feature_dim)
        # It stores the map from cluster/state id to cluster median
        super(MDPnet, self).__init__()
        self.config = config

        # Feature embeddings
        # This will map state id to associate feature
        self.feature = nn.Embedding(self.config.state_size, self.config.feature_dim)
        self.feature.weight.data = feature_table
        for param in self.feature.parameters():
            param.requires_grad = False

        # representation
        mlp_layers = []
        prev_hidden_size = config.feature_dim
        for next_hidden_size in config.rep_hidden_dims:
            mlp_layers.extend([
                nn.Linear(prev_hidden_size, next_hidden_size),
                nn.Tanh(),
            ])
            prev_hidden_size = next_hidden_size
        self.representation = nn.Sequential(*mlp_layers)
        self.rep_dim = prev_hidden_size

        # Transition
        mlp_layers = []
        for next_hidden_size in config.transition_hidden_dims:
            mlp_layers.extend([
                nn.Linear(prev_hidden_size, next_hidden_size),
                nn.Tanh(),
            ])
            prev_hidden_size = next_hidden_size
        mlp_layers.append(
            nn.Linear(prev_hidden_size,config.action_size*config.state_size)
        )
        self.transition = nn.Sequential(*mlp_layers)

        #Reward
        mlp_layers = []
        prev_hidden_size = self.rep_dim
        for next_hidden_size in config.reward_hidden_dims:
            mlp_layers.extend([
                nn.Linear(prev_hidden_size, next_hidden_size),
                nn.Tanh(),
            ])
            prev_hidden_size = next_hidden_size
        mlp_layers.append(
            nn.Linear(prev_hidden_size, config.action_size)
        )
        self.reward = nn.Sequential(*mlp_layers)

    def forward(self,state_id):
        # state = (batch_size x 1) or (batch_size)
        state_id = state_id.data.squeeze() # state: (batch_size, )
        feature = Variable(self.feature_table.index_select(state_id,0))
        rep = self.representation(feature) # rep: (batch_size x rep_dim)
        reward = self.reward(rep).view(-1,self.config.action_size) # reward: (batch_size x 1)
        next_state_prob = self.transition(rep).view(-1,self.config.action_size,self.config.state_size)
        # next_state_prob: (batch_size x action_size x state_size)
        next_state_prob = F.softmax(next_state_prob,2)
        return next_state_prob, reward, rep

