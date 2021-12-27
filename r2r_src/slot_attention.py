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
        self.feature_dim = args.feature_size

        self.gru = nn.GRUCell(dim, self.feature_dim)
        self.mlp = nn.Sequential(
            nn.Linear(self.feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, self.feature_dim)
        )
        self.norm_slots = nn.LayerNorm(self.feature_dim)
        self.norm_pre_ff = nn.LayerNorm(self.feature_dim)
        self.norm_input = nn.LayerNorm(self.feature_dim)

        self.slot_dropout = nn.Dropout(drop_rate)
        self.input_dropout = nn.Dropout(drop_rate)

    def forward(self, cand_feat, pano_feat, cand_mask):
        b, n, d, device = *pano_feat.shape, pano_feat.device

        # original cand_feat as the initial slot
        slots = cand_feat

        pano_feat[..., : -args.angle_feat_size] = self.norm_input(pano_feat[..., : -args.angle_feat_size].clone())
        pano_feat[..., : -args.angle_feat_size] = self.input_dropout(pano_feat[..., : -args.angle_feat_size])


        # (bs, num_ctx, hidden_size)
        k, v = self.to_k(pano_feat), self.to_v(pano_feat)

        for t in range(self.iters):
            slots_prev = slots

            slots[..., : -args.angle_feat_size] = self.norm_slots(slots[..., : -args.angle_feat_size].clone())
            slots[..., : -args.angle_feat_size] = self.slot_dropout(slots[..., :-args.angle_feat_size])

            # (bs, num_slots, hidden_size)
            q = self.to_q(slots.clone())

            # (bs, num_slots, num_ctx)
            dots = torch.einsum('bid,bjd->bij', q, k) * self.scale
            dots.masked_fill_(cand_mask.unsqueeze(-1), -float('inf'))
            attn = dots.softmax(dim=1)
            # attn = dots.softmax(dim=1) + self.eps
            # attn = attn / attn.sum(dim=-1, keepdim=True)

            # (bs, num_slots, hidden_size)
            updates = torch.einsum('bjd,bij->bid', v, attn)

            gru_h = slots_prev.clone()[..., :-args.angle_feat_size].reshape(-1, self.feature_dim)
            gru_updates = self.gru(updates.reshape(-1, d), gru_h)
            gru_updates = gru_updates.reshape(b, -1, gru_updates.shape[-1])
            gru_updates = gru_updates + self.mlp(self.norm_pre_ff(gru_updates))

            slots[...,:-args.angle_feat_size] = gru_updates.clone()
            # slots = gru_updates

        return slots, attn
