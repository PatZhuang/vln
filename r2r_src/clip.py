import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
from param import args
import numpy as np


class Clip(nn.Module):
    def __init__(self, batch_size, input_size, hidden_size, num_layers=1, bidirectional=False):
        super(Clip, self).__init__()
        self.batch_size = batch_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers if not bidirectional else num_layers * 2

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, bidirectional=bidirectional)
        self.W_i = nn.Linear(hidden_size, hidden_size)
        self.W_t = nn.Linear(hidden_size, hidden_size)
        self.LN_i = nn.LayerNorm(hidden_size)
        self.LN_t = nn.LayerNorm(hidden_size)

        self.temperature = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, vis_input, lan_input):
        h_0 = torch.randn(self.num_layers, self.batch_size, self.hidden_size).cuda()
        c_0 = torch.randn(self.num_layers, self.batch_size, self.hidden_size).cuda()

        lengths = torch.tensor([len(v) for v in vis_input]).cuda()
        vis_input = nn.utils.rnn.pad_sequence(vis_input)
        vis_input = nn.utils.rnn.pack_padded_sequence(vis_input, lengths, enforce_sorted=False)
        lstm_output, (h_t, c_t) = self.lstm(vis_input, (h_0, c_0))

        # I_f
        vis_encoding = self.LN_i(self.W_i(h_t.squeeze()))
        # T_f
        lan_encoding = self.LN_t(self.W_t(lan_input))
        # (bs, bs)
        logits = torch.matmul(vis_encoding, lan_encoding.T) * torch.exp(self.temperature)

        return logits
