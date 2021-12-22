import torch
from torch import nn
from torch.nn import init
from param import args


class SlotAttention(nn.Module):
    def __init__(self, num_slots, dim, iters=3, eps=1e-8, hidden_dim=768):
        super().__init__()
        self.num_slots = num_slots
        self.iters = iters
        self.eps = eps
        self.scale = dim ** -0.5

        # self.slots_mu = nn.Parameter(torch.randn(1, 1, dim))
        #
        # self.slots_logsigma = nn.Parameter(torch.zeros(1, 1, dim))
        # init.xavier_uniform_(self.slots_logsigma)

        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)

        self.gru = nn.GRUCell(dim, dim)

        hidden_dim = max(dim, hidden_dim)

        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, dim)
        )

        self.norm_input = nn.LayerNorm(dim)
        self.norm_slots = nn.LayerNorm(dim)
        self.norm_pre_ff = nn.LayerNorm(dim)

        self.dropout = nn.Dropout(0.6)

    def forward(self, cand_feat, pano_feat, cand_mask, dropout=False):
        b, n, d, device = *pano_feat.shape, pano_feat.device

        # n_s = cand_feat.shape[1]

        # mu = self.slots_mu.expand(b, n_s, -1)
        # sigma = self.slots_logsigma.exp().expand(b, n_s, -1)
        #
        # slots = mu + sigma * torch.randn(mu.shape, device=device)

        slots = cand_feat
        pano_feat = self.norm_input(pano_feat)

        if dropout:
            # slots[...,-args.angle_feat_size] = self.dropout(slots[...,-args.angle_feat_size])
            pano_feat[...,:-args.angle_feat_size] = self.dropout(pano_feat[...,:-args.angle_feat_size])

        # original inputs as the initial slot

        # (bs, num_ctx, hidden_size)
        k, v = self.to_k(pano_feat), self.to_v(pano_feat)

        for _ in range(self.iters):
            slots_prev = slots

            slots = self.norm_slots(slots)

            # (bs, num_slots, hidden_size)
            q = self.to_q(slots)

            # (bs, num_slots, num_ctx)
            dots = torch.einsum('bid,bjd->bij', q, k) * self.scale
            dots.masked_fill_(cand_mask.unsqueeze(-1), -float('inf'))
            attn = dots.softmax(dim=1)
            # attn = dots.softmax(dim=1) + self.eps
            # attn = attn / attn.sum(dim=-1, keepdim=True)

            # (bs, num_slots, hidden_size)
            updates = torch.einsum('bjd,bij->bid', v, attn)
            slots = self.gru(
                updates.reshape(-1, d),
                slots_prev.reshape(-1, d)
            )

            slots = slots.reshape(b, -1, d)
            slots = slots + self.mlp(self.norm_pre_ff(slots))

        return slots
