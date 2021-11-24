import numpy as np
import torch
import math
import matplotlib.pyplot as plt
from torch.nn import Module, ModuleList, LSTM, Linear, Parameter, BatchNorm1d
from torch.nn import functional as F
from config import Config
import warnings
import os

warnings.filterwarnings("ignore")
torch.manual_seed(3)

graph_point = ['ih', 'hh', 'i', 'f', 'g', 'o', 'new_c', 'new_h']


def gen_path(path, *paths, **kw):
    store_path = os.path.join(path, *paths)
    if not os.path.exists(store_path):
        os.makedirs(store_path)
    if 'filename' in kw:
        return os.path.join(store_path, kw['filename'])
    else:
        return store_path


def draw_data(data, current_round, date_name):
    num_layers = len(data)
    fig = plt.figure(figsize=(24, 10), facecolor='lightgrey')
    fig.suptitle(t='Round %d, Data of graph point' % current_round, fontsize=20)
    for row in range(num_layers):
        for col in range(len(graph_point)):
            ax = fig.add_subplot(num_layers, len(graph_point), row * len(graph_point) + col + 1)
            # we want to draw layers near the output on the top of the figure, thus data[3-row]
            if row == 0:
                ax.set(xlabel='Time step')
            if row == num_layers-1:
                ax.set(xlabel=graph_point[col])
            if col == 0:
                ax.set(ylabel='LSTM_Layer %d' % (row+1))
    fig.subplots_adjust(hspace=0.3, wspace=0.5)
    num_list = os.listdir(gen_path(Config.path, date_name))
    fig.savefig(gen_path(Config.path, date_name, str(sorted(num_list)[-1]), 'graph', 'point_data',
                         filename='Round%d_data' % current_round + '.png'))
    plt.close()


def draw_grad(grad, current_round, date_name):
    num_layers = len(grad)
    fig = plt.figure(figsize=(24, 10))
    fig.suptitle(t='Round %d, Grad of graph point' % current_round, fontsize=20)
    for row in range(num_layers):
        for col in range(len(graph_point)):
            ax = fig.add_subplot(num_layers, len(graph_point), row*len(graph_point)+col+1)
            # grad is append to list backward, so before plotting we need to reverse the list of grad.
            ax.plot(np.arange(1, Config.time_step + 1), list(reversed(grad[row][graph_point[col]])))
            if row == 0:
                ax.set(xlabel='Time step')
            if row == num_layers-1:
                ax.set(xlabel=graph_point[col])
            if col == 0:
                ax.set(ylabel='LSTM_Layer %d' % (row+1))
    fig.subplots_adjust(hspace=0.3, wspace=0.5)
    num_list = os.listdir(gen_path(Config.path, date_name))
    fig.savefig(gen_path(Config.path, date_name, str(sorted(num_list)[-1]), 'graph', 'point_grad',
                         filename='Round%d_grad' % current_round + '.png'))
    plt.close()


class BNLSTMCell(Module):
    def __init__(self, input_size, hidden_size, normalization_method):
        super(BNLSTMCell, self).__init__()
        self.method = normalization_method
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.W_ih = Parameter(torch.Tensor(self.input_size, self.hidden_size * 4))  # W
        self.W_hh = Parameter(torch.Tensor(self.hidden_size, self.hidden_size * 4))  # U
        self.lbias = Parameter(torch.Tensor(self.hidden_size * 4))
        self.graph_point = ['ih', 'hh', 'i', 'f', 'g', 'o', 'new_c', 'new_h']
        self.data_dict = dict()
        self.grad_dict = dict()
        if self.method == 'none':
            pass
        elif self.method == 'BN':
            affine, stat = False, True
            # self.bn_gate = BatchNorm1d(4 * self.hidden_size, affine=affine, track_running_stats=stat)
            self.bn_ih_list = ModuleList([BatchNorm1d(4 * self.hidden_size, affine=affine, track_running_stats=stat)
                                          for _ in range(Config.time_step)])
            self.bn_ih_gamma = Parameter(torch.Tensor(4 * self.hidden_size))
            self.bn_hh_list = ModuleList([BatchNorm1d(4 * self.hidden_size, affine=affine, track_running_stats=stat)
                                          for _ in range(Config.time_step)])
            self.bn_hh_gamma = Parameter(torch.Tensor(4 * self.hidden_size))
            self.bn_c_list = ModuleList([BatchNorm1d(self.hidden_size, affine=affine, track_running_stats=stat)
                                         for _ in range(Config.time_step)])
            self.bn_c_gamma = Parameter(torch.Tensor(self.hidden_size))
            self.bn_c_beta = Parameter(torch.Tensor(self.hidden_size))
        self.init_parameters()

    def init_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for name, param in self.named_parameters():
            if 'W_ih' in name:
                for w_ih in param.data.chunk(4, dim=1):
                    torch.nn.init.xavier_uniform_(w_ih.data)
                    # print name+' initialized'
            elif 'W_hh' in name:
                for w_hh in param.data.chunk(4, dim=1):
                    torch.nn.init.orthogonal_(w_hh.data)
                    # print name + ' initialized'
                    # print torch.mm(w_hh.T, w_hh)  = identity matrix
            elif 'lbias' in name:
                torch.nn.init.uniform_(param, -stdv, stdv)
                # param.data.fill_(1)
                param.data.chunk(4)[1].fill_(1)
                # print name + ' initialized'
            elif 'gamma' in name:
                torch.nn.init.ones_(param)
            elif 'beta' in name:
                torch.nn.init.zeros_(param)

    def reset_data_dict(self):
        self.data_dict = dict(zip(self.graph_point, [[] for _ in range(len(self.graph_point))]))

    def reset_grad_dict(self):
        self.grad_dict = dict(zip(self.graph_point, [[] for _ in range(len(self.graph_point))]))

    def hook_ih(self, grad):
        self.grad_dict['ih'].append(grad.mean().item())

    def hook_hh(self, grad):
        self.grad_dict['hh'].append(grad.mean().item())

    def hook_i(self, grad):
        self.grad_dict['i'].append(grad.mean().item())

    def hook_f(self, grad):
        self.grad_dict['f'].append(grad.mean().item())

    def hook_g(self, grad):
        self.grad_dict['g'].append(grad.mean().item())

    def hook_o(self, grad):
        self.grad_dict['o'].append(grad.mean().item())

    def hook_new_c(self, grad):
        self.grad_dict['new_c'].append(grad.mean().item())

    def hook_new_h(self, grad):
        self.grad_dict['new_h'].append(grad.mean().item())

    def forward(self, input, h, c, current_step, require_point_data, require_point_grad):
        '''
        :param input: x at one time_step, with shape of (batch_size, input_size)
        :param hx: (h, c) two hidden states of a lstm cell
        :return:
        '''
        ih = torch.matmul(input, self.W_ih)  # (batch_size, hidden_size * 4) = (batch_size, input_size) dot (input_size, hidden_size * 4)
        hh = torch.matmul(h, self.W_hh)  # (batch_size, hidden_size * 4) = (batch_size, hidden_size) dot (hidden_size, hidden_size * 4)
        if self.method == 'BN':
            ih = self.bn_ih_list[current_step].forward(ih)
            ih = torch.mul(ih, self.bn_ih_gamma)
            hh = self.bn_hh_list[current_step].forward(hh)
            hh = torch.mul(hh, self.bn_hh_gamma)
        gates = ih + hh + self.lbias
        i, f, g, o = gates.chunk(4, dim=1)
        new_c = torch.mul(torch.sigmoid(f), c) + torch.mul(torch.sigmoid(i), torch.tanh(g))
        if self.method == 'BN':
            new_c = self.bn_c_list[current_step].forward(new_c)
            new_c = torch.mul(new_c, self.bn_c_gamma) + self.bn_c_beta
        new_h = torch.mul(torch.sigmoid(o), torch.tanh(new_c))  # (batch_size, hidden_size)
        if require_point_data:
            self.data_dict['ih'].append(ih.mean().item())
            self.data_dict['hh'].append(hh.mean().item())
            self.data_dict['i'].append(i.mean().item())
            self.data_dict['f'].append(f.mean().item())
            self.data_dict['g'].append(g.mean().item())
            self.data_dict['o'].append(o.mean().item())
            self.data_dict['new_c'].append(new_c.mean().item())
            self.data_dict['new_h'].append(new_h.mean().item())
        if require_point_grad:
            ih.register_hook(self.hook_ih)
            hh.register_hook(self.hook_hh)
            i.register_hook(self.hook_i)
            f.register_hook(self.hook_f)
            g.register_hook(self.hook_g)
            o.register_hook(self.hook_o)
            new_c.register_hook(self.hook_new_c)
            new_h.register_hook(self.hook_new_h)
        return new_h, new_c


class BNLSTM(Module):
    def __init__(self, input_size, hidden_size, dropout_rate, date_name, normalization_method='BN'):
        super(BNLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.method = normalization_method
        self.current_round = 0
        self.date_name = date_name
        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.lstm_cell_1 = BNLSTMCell(input_size, hidden_size, self.method)
        self.lstm_cell_2 = BNLSTMCell(hidden_size, hidden_size, self.method)
        #self.lstm_cell_3 = BNLSTMCell(hidden_size, hidden_size, self.method)


    def init_hidden(self, bs, device):
        init = False
        h1, c1 = torch.zeros(bs, self.hidden_size), torch.zeros(bs, self.hidden_size)
        h2, c2 = torch.zeros(bs, self.hidden_size), torch.zeros(bs, self.hidden_size)
        #h3, c3 = torch.zeros(bs, self.hidden_size), torch.zeros(bs, self.hidden_size)
        if init:
            torch.nn.init.xavier_normal_(h1)
            torch.nn.init.xavier_normal_(c1)
            torch.nn.init.xavier_normal_(h2)
            torch.nn.init.xavier_normal_(c2)
            torch.nn.init.xavier_normal_(h3)
            torch.nn.init.xavier_normal_(c3)
        h1, c1 = (h1.to(device), c1.to(device))
        h2, c2 = (h2.to(device), c2.to(device))
        #h3, c3 = (h3.to(device), c3.to(device))

        return h1, h2, c1, c2

    def forward(self, x, hc=None):
        bs, seq_len, features = x.size()
        if hc is None:
            h1, h2, c1, c2 = self.init_hidden(bs, x.device)
        hidden_seq = []
        require_point_data, require_point_grad = False, False
        if self.training:
            '''so actually the point data & grad to draw are from train batch'''
            self.current_round += 1  # just to synchronize with the main training process, to be passed to draw figures
            if self.current_round % Config.eval_interval == 1:
                if self.current_round >= Config.eval_interval:  # at 1st round there is no grad data of previous round
                    '''Draw grad of last eval round'''
                    grad = [self.lstm_cell_1.grad_dict, self.lstm_cell_2.grad_dict]
                    draw_grad(grad, self.current_round-Config.eval_interval, self.date_name)
                require_point_data, require_point_grad = True, True  # tell LSTM_Cell to record data and register hook
                self.lstm_cell_1.reset_data_dict()  # clear data and grad dict, prepared for incoming new round
                self.lstm_cell_1.reset_grad_dict()
                self.lstm_cell_2.reset_data_dict()
                self.lstm_cell_2.reset_grad_dict()
                #self.lstm_cell_3.reset_data_dict()
                #self.lstm_cell_3.reset_grad_dict()
        for time_step in range(seq_len):
            h1, c1 = self.lstm_cell_1(x[:, time_step, :], h1, c1, time_step, require_point_data, require_point_grad)
            h2, c2 = self.lstm_cell_2(self.dropout(h1), h2, c2, time_step, require_point_data, require_point_grad)
            # h3, c3 = self.lstm_cell_3(self.dropout(h2), h3, c3, require_point_data, require_point_grad)
            hidden_seq.append(h2.unsqueeze(0))
        hidden_seq = torch.cat(hidden_seq, dim=0)
        # reshape from shape (sequence, batch, feature) to (batch, sequence, feature)
        hidden_seq = hidden_seq.transpose(0, 1).contiguous()
        # print 'BNLSTMforward', self.current_round, self.date_name, self.training
        '''Draw data of current eval round'''
        if require_point_data:
            data = [self.lstm_cell_1.data_dict, self.lstm_cell_2.data_dict]
            draw_data(data, self.current_round, self.date_name)
        return hidden_seq


if __name__ == '__main__':
    device = torch.device("cuda: 1")
    lstm = BNLSTM(input_size=10, hidden_size=1, date_name='a', dropout_rate=0.0, normalization_method='BN')
    lstm.to(device)
    print list(lstm.lstm_cell_1.bn_c_list[0].named_parameters())
    for name, param in lstm.named_parameters():
        print name, param.size(), param.requires_grad
    '''  
    x = torch.randn(3, 60, 10).to(device)
    lstm_out = lstm(x)
    lstm_out = lstm_out.cpu()
    y = lstm_out * 2
    loss_fn = torch.nn.MSELoss()
    loss = loss_fn(lstm_out, y)
    print loss, loss.size()
    '''