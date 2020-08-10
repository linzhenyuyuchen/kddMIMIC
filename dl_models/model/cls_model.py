import os
import sys
import warnings
import numpy as np
from tqdm import tqdm
from random import choice
from copy import deepcopy
from string import ascii_uppercase

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

warnings.filterwarnings('ignore')

class newLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(newLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.layer = nn.Sequential(nn.Linear(self.in_features, self.out_features),
                                   nn.Sigmoid(),
                                   nn.Dropout(0.2))
    def forward(self, x):
        out = self.layer(x)
        return out

# Simple feed-forward network, used to get network outputs and evaluations
# Input: 0-mean, 1-std, 2-dimension array (n_samples, n_dimensions)
# Output: 0/1 binary label (n_samples)
# Layers: 2 hidden layers and 1 prediction layer
# Activation: sigmoid for all layers
# Objective: binary cross entropy for prediction
class FeedForwardNetwork(nn.Module):
    def __init__(self, n_features=5, hidden_dim=10, patience=20, ffn_depth=2,batch_normalization='False'):
        super(FeedForwardNetwork, self).__init__()

        self.n_features = n_features
        self.hidden_dim = hidden_dim
        self.patience = patience
        self.type_MMDL = 1
        self.random_str = ''.join(choice(ascii_uppercase) for i in range(12))
        self.ffn_depth = ffn_depth
        self.batch_normalization = batch_normalization

        # input n_features
        self.linear1 = nn.Sequential(nn.Linear(self.n_features, self.hidden_dim),
                                     nn.Sigmoid(),
                                     nn.Dropout(0.1))
        self.layers = nn.Sequential()
        for t in range(self.ffn_depth - 1):
            self.layers.add_module(f"layer{t}",newLayer(self.hidden_dim, self.hidden_dim))
        self.linear2 = nn.Linear(self.hidden_dim, 1)
        self.bn = nn.BatchNorm1d(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.linear1(x)
        x = self.layers(x)
        x = self.linear2(x)
        if self.batch_normalization:
            x = self.bn(x)
        x = self.sigmoid(x)
        return x

# Simple LSTM network with a overall prediction layer
# Input: 0-mean, 1-std, 3-dimension array (n_samples, n_timesteps, n_dimensions)
# Output: 0/1 binary label    (n_samples)
# Layers: 2 hidden layers and 1 prediction layer
# Activation: sigmoid for all layers
# Objective: binary cross entropy for prediction
class SimpleLSTMNetwork(nn.Module):
    def __init__(self, n_features=200, time_step = 24, lstm_layers=1):
        super(SimpleLSTMNetwork, self).__init__()
        self.n_features = n_features
        self.lstm_layers = lstm_layers
        self.time_step = time_step
        self.lstm = nn.LSTM(input_size = self.n_features, hidden_size = self.n_features, num_layers=self.lstm_layers, batch_first=True)
        self.model = nn.Sequential(
            nn.Sigmoid(),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(self.n_features * self.time_step , 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.model(x)
        return x



class HierarchicalMultimodal(nn.Module):
    def __init__(self, static = True, size_Xs= 5, dropout = 0.1, batch_normalization = 'False',
                 time_step = 48, n_features = 15, fit_parameters = [2,1,0]):
        super(HierarchicalMultimodal, self).__init__()

        self.static = static
        self.size_Xs = size_Xs
        self.dropout = dropout
        self.batch_normalization = batch_normalization
        self.n_features = n_features
        self.time_step = time_step
        self.output_dim,self.static_depth, self.merge_depth = fit_parameters
        self.y_tasks = 1
        ########################################################################
        if self.static:
            # FFN model
            self.FFNmodel = nn.Sequential()
            self.FFNmodel.add_module("linear1", nn.Linear(self.size_Xs, self.size_Xs * self.output_dim))
            self.FFNmodel.add_module("sg1", nn.Sigmoid())
            self.FFNmodel.add_module("dp1", nn.Dropout(self.dropout))

            for i in range(self.static_depth):
                self.FFNmodel.add_module(f"linear{i+2}",nn.Linear(self.size_Xs * self.output_dim, self.size_Xs * self.output_dim))
                self.FFNmodel.add_module(f"sg{i+2}",nn.Sigmoid())
                self.FFNmodel.add_module(f"dp{i+2}",nn.Dropout(self.dropout))

        ########################################################################

        self.GRUmodel = nn.Sequential()
        self.GRUmodel.add_module("gru", nn.GRU(input_size= self.n_features, hidden_size= self.n_features * self.output_dim, batch_first=True))
        self.GRUlast = nn.Sequential()
        self.GRUlast.add_module("flatten", nn.Flatten())
        self.GRUlast.add_module("dropout", nn.Dropout(self.dropout))


        if self.static:
            self.len_combine = self.size_Xs + self.n_features
            self.model_merge = nn.Sequential(nn.Linear(self.size_Xs * self.output_dim+self.time_step * self.n_features * self.output_dim, self.len_combine*self.output_dim),
                                       nn.Sigmoid(),
                                       nn.Dropout(self.dropout))

            for i in range(self.merge_depth):
                dense_len = int(self.len_combine * self.output_dim / np.power(2, i + 1))
                if i == 0:
                    self.model_merge.add_module(f"mlinear{i}",nn.Linear(self.len_combine*self.output_dim, dense_len))
                else:
                    self.model_merge.add_module(f"mlinear{i}", nn.Linear(dense_len, dense_len))
                self.model_merge.add_module(f"sg{i}",nn.Sigmoid())
                self.model_merge.add_module(f"dp{i}",nn.Dropout(self.dropout))
            if self.merge_depth:
                merge_input_len = int(self.len_combine * self.output_dim / np.power(2, self.merge_depth))
            else:
                merge_input_len =  self.len_combine*self.output_dim
            self.model_merge.add_module("mlinear_last", nn.Linear(merge_input_len, self.y_tasks))

            if self.batch_normalization:
                self.model_merge.add_module("bn", nn.BatchNorm1d(self.y_tasks))
            self.model_merge.add_module("sg_last", nn.Sigmoid())

        else:
            self.len_combine = self.n_features * self.output_dim
            self.model_merge = Sequential()
            self.model_merge.add_module("mlinear_last", nn.Linear(self.len_combine, self.y_tasks))
            if self.batch_normalization:
                self.model_merge.add_module("bn", nn.BatchNorm1d(self.y_tasks))
            self.model_merge.add_module("sg_last", nn.Sigmoid())


    def forward(self, x):
        if self.static:
            x0, x1 = x[0], x[1]
            x0 = self.FFNmodel(x0)
            x1,_ = self.GRUmodel(x1)
            x1 = self.GRUlast(x1)
            x = torch.cat([x0, x1], dim=1)
            x = self.model_merge(x)
        else:
            x = self.GRUmodel(x)
            x = self.model_merge(x)

        return x




