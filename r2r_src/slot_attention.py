import torch
from torch import nn
from torch.nn import init
from param import args


class SlotAttention(nn.Module):
    def __init__(self, num_slots, dim, iters=3, eps=1e-8, hidden_dim=768, drop_rate=0.4):
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

        hidden_dim = max(dim, hidden_dim)

        if args.slot_ignore_angle:
            self.gru = nn.GRUCell(dim, args.feature_size)
            self.mlp = nn.Sequential(
                nn.Linear(args.feature_size, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, args.feature_size)
            )
            self.norm_slots = nn.LayerNorm(args.feature_size)
            self.norm_pre_ff = nn.LayerNorm(args.feature_size)
        else:
            self.gru = nn.GRUCell(dim, dim)
            self.mlp = nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, dim)
            )
            self.norm_slots = nn.LayerNorm(dim)
            self.norm_pre_ff = nn.LayerNorm(dim)

        self.norm_input = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, cand_feat, pano_feat, cand_mask):
        b, n, d, device = *pano_feat.shape, pano_feat.device

        # n_s = cand_feat.shape[1]

        # mu = self.slots_mu.expand(b, n_s, -1)
        # sigma = self.slots_logsigma.exp().expand(b, n_s, -1)
        #
        # slots = mu + sigma * torch.randn(mu.shape, device=device)

        # original cand_feat as the initial slot
        slots = cand_feat
        pano_feat[:-args.angle_feat_size] = self.norm_input(pano_feat[:-args.angle_feat_size].clone())

        # slots[...,:-args.angle_feat_size] = self.dropout(slots[...,:-args.angle_feat_size])
        pano_feat[...,:-args.angle_feat_size] = self.dropout(pano_feat[...,:-args.angle_feat_size])

        # (bs, num_ctx, hidden_size)
        k, v = self.to_k(pano_feat), self.to_v(pano_feat)

        for _ in range(self.iters):
            slots_prev = slots
            slots[...,:-args.angle_feat_size] = self.norm_slots(slots[...,:-args.angle_feat_size].clone())

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

            if args.slot_ignore_angle:
                gru_h = slots_prev.reshape(-1, d)[...,:-args.angle_feat_size].clone()
            else:
                gru_h = slots_prev.reshape(-1, d)

            gru_updates = self.gru(updates.reshape(-1, d), gru_h)
            gru_updates = gru_updates.reshape(b, -1, gru_updates.shape[-1])
            gru_updates = gru_updates + self.mlp(self.norm_pre_ff(gru_updates))

            if args.slot_ignore_angle:
                slots[...,:-args.angle_feat_size] = gru_updates.clone()
            else:
                slots = gru_updates

        return slots
