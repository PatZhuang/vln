import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from param import args


class PGMonitor(nn.Module):
    def __init__(self, batch_size=8, hidden_size=768):
        super(PGMonitor, self).__init__()
        self.batch_size = batch_size
        self.hidden_size = hidden_size

        self.shift_h_t = torch.zeros(batch_size, hidden_size)
        self.shift_c_t = torch.zeros(batch_size, hidden_size)
        self.shift_instr_lstm = nn.LSTMCell(hidden_size, hidden_size)
        self.shift_instr_FC = nn.Sequential(
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )

        self.instr_pg_FC = nn.Linear(hidden_size, 1)
        self.path_pg_FC = nn.Linear(hidden_size, 1)

    def forward(self, state):
        self.shift_h_t, self.shift_c_t = self.shift_instr_lstm(state)
        shift_pred = self.shift_instr_FC(self.shift_h_t)

        instr_pg_pred = self.instr_pg_FC(state)
        path_pg_pred = self.path_pg_FC(state)

        return shift_pred, instr_pg_pred, path_pg_pred


