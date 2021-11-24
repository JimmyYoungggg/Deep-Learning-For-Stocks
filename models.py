import numpy as np
import pandas as pd
import os
import torch
import gc
import shap
import multiprocessing as mp
from collections import OrderedDict
from datetime import datetime
from torch import Tensor
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import init
from torch.nn.modules import Module
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from config import Config, TradingDay, gen_path
from sklearn.preprocessing import StandardScaler
import joblib
from torch.nn.parameter import Parameter
import math
import torch
import torch.nn as nn
import numpy as np
import time
import math
from matplotlib import pyplot
import copy

torch.manual_seed(0)
np.random.seed(0)


class MLP(nn.Module):
    def __init__(self, config):
        super(MLP, self).__init__()
        self.config = config
        affine, stat = True, True
        layers = []
        input_size = config.input_size
        for i, hidden_size in enumerate(
                config.hidden_size_mlp):  # so the num of hidden layers depends on len(config.hidden_size_mlp)
            layer = [('Dense_%d' % (i + 1), nn.Linear(input_size, hidden_size)),
                     ('Relu_%d' % (i + 1), nn.ReLU()),
                     ('BN_%d' % (i + 1), nn.BatchNorm1d(hidden_size, affine=affine, track_running_stats=stat)),
                     ('Drop_%d' % (i + 1), nn.Dropout(p=self.config.dropout_rate))]
            input_size = hidden_size
            layers.extend(layer)
        output_layer = [('Output_Dense', nn.Linear(input_size, config.output_size)),
                        ('Output_BN', nn.BatchNorm1d(config.output_size, affine=affine, track_running_stats=stat))]
        layers.extend(output_layer)
        self.layer = torch.nn.Sequential(OrderedDict(layers))

    def forward(self, x, output_feature=False):
        x = x.view(x.shape[0], -1)
        prediction = self.layer(x)  # .squeeze()
        # x (batch_size, 1, input_size) view to (batch_size, input_size); prediction (batch_size, 1)
        if output_feature:
            return prediction, None
        else:
            return prediction


class MLPwithGAN(nn.Module):
    def __init__(self, config):
        super(MLPwithGAN, self).__init__()
        self.config = config
        affine, stat = True, True
        layers = []
        input_size = config.input_size

        for i, hidden_size in enumerate(
                config.hidden_size_mlp):  # so the num of hidden layers depends on len(config.hidden_size_mlp)
            layer = [('Dense_%d' % (i + 1), nn.Linear(input_size, hidden_size)),
                     ('Relu_%d' % (i + 1), nn.ReLU()),
                     ('BN_%d' % (i + 1), nn.BatchNorm1d(hidden_size, affine=affine, track_running_stats=stat)),
                     ('Drop_%d' % (i + 1), nn.Dropout(p=self.config.dropout_rate))]
            input_size = hidden_size
            layers.extend(layer)
        self.layer = torch.nn.Sequential(OrderedDict(layers))

        output_layer_pred = [('Output_Dense', nn.Linear(input_size, config.output_size)),
                             ('Output_BN', nn.BatchNorm1d(config.output_size, affine=affine, track_running_stats=stat))]
        self.feature_extraction = torch.nn.Sequential(OrderedDict(output_layer_pred))

    def forward(self, x, output_feature=False):
        x = x.view(x.shape[0], -1)
        x = self.layer(x)  # .squeeze()
        prediction = self.feature_extraction(x)
        # x (batch_size, input_size); prediction (batch_size, 1)
        if output_feature:
            return prediction, x
        else:
            return prediction


class GAN(nn.Module):
    def __init__(self, config):
        super(GAN, self).__init__()
        self.config = config
        affine, stat = True, True
        input_size = config.hidden_size_mlp[-1]

        output_layer_pred = [('Output_Dense', nn.Linear(input_size, config.output_size)),
                             ('Output_BN', nn.BatchNorm1d(config.output_size, affine=affine, track_running_stats=stat))]
        self.feature_extraction = torch.nn.Sequential(OrderedDict(output_layer_pred))

    def forward(self, x):
        prediction = self.feature_extraction(x)
        # x (batch_size, input_size); prediction (batch_size, 1)
        return prediction


class VanillaLSTM(nn.Module):
    def __init__(self, config):
        super(VanillaLSTM, self).__init__()
        self.config = config
        self.hidden_size = config.hidden_size_lstm
        self.lstm_layers = nn.ModuleList()
        input_size = config.input_size
        for hidden_size in config.hidden_size_lstm:
            self.lstm_layers.append(
                nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=1, batch_first=True))
            input_size = hidden_size
        self.linear = nn.Linear(input_size, self.config.output_size)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if type(m) in [torch.nn.GRU, torch.nn.LSTM, torch.nn.RNN]:
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        for w_ih in param.data.chunk(4):
                            torch.nn.init.xavier_uniform_(w_ih.data)
                    elif 'weight_hh' in name:
                        for w_hh in param.data.chunk(4):
                            torch.nn.init.orthogonal_(w_hh.data)
                            # print torch.mm(w_hh.T, w_hh)  = identity matrix
                    elif 'bias' in name:
                        # param.data.fill_(1)
                        param.data.chunk(4)[1].fill_(1)

    def forward(self, x, hidden=None):
        for lstm_layer in self.lstm_layers:
            x, _ = lstm_layer(x, hidden)
            x = F.dropout(x, p=self.config.dropout_rate, training=self.training)
        # x.size (batch_size, time_step, input_size); lstm_out.size (batch_size, time_step, hidden_size)
        x = x[:, -1, :]
        linear_out = self.linear(x)  # .squeeze()
        # linear_out size (batch_size, 1)
        return linear_out


class CNNfeature(Module):
    def __init__(self, config):
        super(CNNfeature, self).__init__()
        self.config = config
        self.conv1 = nn.Sequential(         # input shape (batch_size, 1, time_step, input_size)
            nn.Conv2d(
                in_channels=1,              # input channels
                out_channels=config.n_filter,   # n_filters
                kernel_size=(1, config.input_size),              # filter size
                stride=1,                   # filter movement/step
                padding=0,                  # if want same width and length of this image after Conv2d, padding=(kernel_size-1)/2 if stride=1
            ),                              # output shape (batch_size, n_filter, time_step, 1)
            nn.BatchNorm2d(config.n_filter),
            nn.ReLU(),                      # activation
            #nn.MaxPool2d(kernel_size=2),    # choose max value in 2x2 area, output shape (16, 14, 14)
        )
        conv_out_size = config.n_filter * config.time_step
        linear_layers = [('Dense_in', nn.Linear(conv_out_size, int(conv_out_size / 4))),
                         ('Relu', nn.ReLU()),
                         ('BN', nn.BatchNorm1d(int(conv_out_size / 4), affine=True, track_running_stats=True)),
                         ('Drop', nn.Dropout(p=self.config.dropout_rate)),
                         ('Dense_out', nn.Linear(int(conv_out_size / 4), config.output_size))]
        self.linear_layers = torch.nn.Sequential(OrderedDict(linear_layers))

    def forward(self, x):
        conv_in = x.unsqueeze(1)
        conv_out = self.conv1(conv_in)
        linear_in = conv_out.view(x.size(0), -1)    # flatten the output of conv2 to (batch_size, n_filter * time_step)
        linear_out = self.linear_layers(linear_in)
        return linear_out