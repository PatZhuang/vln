import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from param import args


class PGMonitor(nn.Module):
    def __init__(self, batch_size=8, hidden_size=256):
        super(PGMonitor, self).__init__()
        self.batch_size = batch_size
        self.hidden_size = hidden_size

        # output (maxInput - 2) probs indicating if that word should be attended
        # self.instr_attn_layer = nn.Sequential(
        #     nn.Linear(hidden_size, hidden_size // 2),
        #     nn.Linear(hidden_size // 2, args.maxInput - 1),
        # )
        self.instr_attn_layer = nn.Sequential(
            nn.Linear(args.maxInput - 1, hidden_size),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, state, mask=None):
        instr_attn_pred = self.instr_attn_layer(state)
        # instr_attn_pred.masked_fill_(mask, -float('inf'))
        instr_attn_pred = F.sigmoid(instr_attn_pred)

        return instr_attn_pred


