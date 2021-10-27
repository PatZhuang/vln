# Recurrent VLN-BERT, 2020, by Yicong.Hong@anu.edu.au

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from param import args
import math

from vlnbert.vlnbert_init import get_vlnbert_models


class NeRF_PE(nn.Module):
    def __init__(self, hidden_size):
        super(NeRF_PE, self).__init__()
        self.hidden_size = hidden_size

    def forward(self, x):
        input_shape = x.shape
        if input_shape[-1] == 2:    # heading, elevation
            x = x.view(-1, 2)
            pe = torch.tensor([[[math.sin(2 ** L * math.pi * pos[0]),
                                 math.cos(2 ** L * math.pi * pos[0]),
                                 math.sin(2 ** L * math.pi * pos[1]),
                                 math.cos(2 ** L * math.pi * pos[1]),
                                 ] for L in range(4)] * (self.hidden_size // 16) for pos in x])

        elif input_shape[-1] == 4:  # bbox
            x = x.view(-1, 4)
            pe = torch.tensor([[[math.sin(2 ** L * math.pi * bbox[0]),
                                          math.cos(2 ** L * math.pi * bbox[0]),
                                          math.sin(2 ** L * math.pi * bbox[1]),
                                          math.cos(2 ** L * math.pi * bbox[1]),
                                          math.sin(2 ** L * math.pi * bbox[2]),
                                          math.cos(2 ** L * math.pi * bbox[2]),
                                          math.sin(2 ** L * math.pi * bbox[3]),
                                          math.cos(2 ** L * math.pi * bbox[3])
                                          ] for L in range(8)] * (self.hidden_size // 64) for bbox in x])
        else:
            raise ValueError('wrong feat size')
        pe = pe.view(list(input_shape[:-1]) + [self.hidden_size])
        return pe

class VLNBERT(nn.Module):
    def __init__(self, feature_size=2048+128):
        super(VLNBERT, self).__init__()
        print('\nInitalizing the VLN-BERT model ...')

        self.vln_bert = get_vlnbert_models(args, config=None)  # initialize the VLN-BERT
        self.vln_bert.config.directions = 4  # a preset random number

        hidden_size = self.vln_bert.config.hidden_size
        layer_norm_eps = self.vln_bert.config.layer_norm_eps

        self.action_state_project = nn.Sequential(
            nn.Linear(hidden_size+args.angle_feat_size, hidden_size), nn.Tanh())
        self.action_LayerNorm = BertLayerNorm(hidden_size, eps=layer_norm_eps)

        self.drop_env = nn.Dropout(p=args.featdropout)
        self.img_projection = nn.Linear(feature_size, hidden_size, bias=True)
        self.cand_LayerNorm = BertLayerNorm(hidden_size, eps=layer_norm_eps)

        self.vis_lang_LayerNorm = BertLayerNorm(hidden_size, eps=layer_norm_eps)
        self.state_proj = nn.Linear(hidden_size*2, hidden_size, bias=True)
        self.state_LayerNorm = BertLayerNorm(hidden_size, eps=layer_norm_eps)

        # object
        self.obj_pos_proj = nn.Linear(4, hidden_size)
        self.cand_pos_proj = nn.Linear(4, hidden_size)
        self.pos_encoding_ln = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.obj_dropout = nn.Dropout(p=args.featdropout)

        if args.max_pool_feature:
            self.feat_cat_fc = nn.Linear(args.feature_size * 2, args.feature_size)
            self.feat_cat_alpha = nn.Parameter(torch.ones(1))

    def forward(self, mode, sentence, token_type_ids=None,
                attention_mask=None, lang_mask=None, vis_mask=None,
                position_ids=None, action_feats=None, pano_feats=None, cand_feats=None,
                cand_pos=None, cand_mask=None, obj_feat=None, obj_bbox=None, cand_mp_feats=None):

        if mode == 'language':
            init_state, encoded_sentence = self.vln_bert(mode, sentence, attention_mask=attention_mask, lang_mask=lang_mask,)
            return init_state, encoded_sentence

        if mode == 'object':
            if not args.nerf_pe:
                # object position in candidate views respectively
                obj_pos_encoding = self.obj_pos_proj(obj_bbox)  # (bs, max_cand_len, 8, hidden_size)
                # candidate position relative to current base view
                cand_pos_encoding = self.cand_pos_proj(cand_pos).unsqueeze(2).repeat(1,1,args.top_N_obj,1)
            else:
                obj_pos_encoding = obj_bbox.cuda()
                cand_pos_encoding = cand_pos.unsqueeze(2).repeat(1,1,args.top_N_obj,1).cuda()
            pos_encoding = self.pos_encoding_ln(obj_pos_encoding + cand_pos_encoding)

            if args.drop_obj:
                obj_feat = self.obj_dropout(obj_feat)

            obj_action_scores = self.vln_bert(mode, sentence, lang_mask=lang_mask, cand_mask=cand_mask,
                                             obj_feat=obj_feat.long(), obj_pos_encoding=pos_encoding)
            return obj_action_scores

        elif mode == 'visual':

            state_action_embed = torch.cat((sentence[:,0,:], action_feats), 1)
            state_with_action = self.action_state_project(state_action_embed)
            state_with_action = self.action_LayerNorm(state_with_action)
            state_feats = torch.cat((state_with_action.unsqueeze(1), sentence[:,1:,:]), dim=1)

            if cand_mp_feats is not None:
                if args.mix_type == 'fc':
                    cand_feats[..., :-args.angle_feat_size] = self.feat_cat_fc(
                        torch.cat((cand_feats[..., :-args.angle_feat_size], cand_mp_feats), -1)
                    )
                elif args.mix_type == 'alpha':
                    cand_feats[..., :-args.angle_feat_size] += self.feat_cat_alpha * cand_mp_feats

            cand_feats[..., :-args.angle_feat_size] = self.drop_env(cand_feats[..., :-args.angle_feat_size])

            # logit is the attention scores over the candidate features
            h_t, logit, attended_language, attended_visual = self.vln_bert(mode, state_feats,
                attention_mask=attention_mask, lang_mask=lang_mask, vis_mask=vis_mask, img_feats=cand_feats)

            # update agent's state, unify history, language and vision by elementwise product
            vis_lang_feat = self.vis_lang_LayerNorm(attended_language * attended_visual)
            state_output = torch.cat((h_t, vis_lang_feat), dim=-1)
            state_proj = self.state_proj(state_output)
            state_proj = self.state_LayerNorm(state_proj)

            return state_proj, logit

        else:
            ModuleNotFoundError


class BertLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(BertLayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.state2value = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(512, 1),
        )

    def forward(self, state):
        return self.state2value(state).squeeze()
