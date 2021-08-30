"""
Reasoning Visual Dialog with Sparse Graph Learning and Knowledge Transfer
Gi-Cheon Kang, Junseok Park, Hwaran Lee, Byoung-Tak Zhang, Jin-Hwa Kim
https://arxiv.org/abs/2004.06698
"""
import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
from visdialch.utils import DynamicRNN
from .module import NodeEmbeddingModule, SparseGraphLearningModule
from .net_utils import FC, MLP, LayerNorm
from torch.nn.utils.weight_norm import weight_norm

class SGLN(nn.Module):
    def __init__(self, __C, vocabulary):
        super().__init__()
        self.__C = __C

        self.word_embed = nn.Embedding(
            num_embeddings=len(vocabulary), 
            embedding_dim=__C["word_embedding_size"], 
            padding_idx=vocabulary.PAD_INDEX
        )

        self.h_rnn = nn.LSTM(
            input_size=__C["word_embedding_size"],
            hidden_size=__C["hidden_size"],
            num_layers=__C["lstm_num_layers"],
            batch_first=True,
        )
        self.q_rnn = nn.LSTM(
            input_size=__C["word_embedding_size"],
            hidden_size=__C["hidden_size"],
            num_layers=__C["lstm_num_layers"],
            batch_first=True,
        )

        self.v_proj = nn.Linear(
            __C["img_feature_size"], 
            __C["hidden_size"]
        )
        self.j_proj = nn.Linear(
            __C["hidden_size"], 
            __C["hidden_size"]
        )

        self.q_norm = LayerNorm(__C['flat_out_size'])
        self.h_norm = LayerNorm(__C['flat_out_size'])

        self.q_attflat_lang = AttFlat(__C)
        self.q_attflat_img = AttFlat(__C)

        self.h_attflat_lang = AttFlat(__C)
        self.h_attflat_img = AttFlat(__C)

        self.q_nem = NodeEmbeddingModule(__C)
        self.h_nem = NodeEmbeddingModule(__C)
        self.sgm = SparseGraphLearningModule(__C)

    def make_mask(self, feature):
        return (torch.sum(
            torch.abs(feature),
            dim=-1
        ) == 0).unsqueeze(1).unsqueeze(2)    
    
    def lang_emb(self, seq, lang_type='ques'):
        rnn_cls = None 
        if lang_type == 'hist':
            rnn_cls = self.h_rnn
        if lang_type == 'ques':
            rnn_cls = self.q_rnn

        lang_feat_mask = self.make_mask(seq.unsqueeze(2))        
        lang_feat = self.word_embed(seq)

        rnn_cls.flatten_parameters()
        lang_feat, _ = rnn_cls(lang_feat)
        return lang_feat, lang_feat_mask

    def forward(self, batch):
        q = batch["ques"]      # b x 10 x 20
        h = batch["hist"]      # b x 10 x 40
        v = batch["img_feat"]  # b x max_obj x 2048

        img_feat = self.v_proj(v) # b x 36 x 1024
        img_feat_mask = self.make_mask(img_feat)

        n_batch, n_round, _ = q.size()
        enc_outs = []
        binary_strct = []
        weighted_strct = Variable(torch.zeros(n_batch, 1, n_round+1).cuda())
        h_embs = None

        for i in range(n_round):
            ques_feat, ques_feat_mask = self.lang_emb(q[:, i, :], 'ques') 
            hist_feat, hist_feat_mask = self.lang_emb(h[:, i, :], 'hist')

            # question node embedding
            q_emb, qi_emb = self.q_nem(
                ques_feat,
                img_feat,
                ques_feat_mask,
                img_feat_mask
            )

            q_emb = self.q_attflat_lang(
                q_emb,
                ques_feat_mask
            )

            qi_emb = self.q_attflat_img(
                qi_emb,
                img_feat_mask
            )

            q_emb = self.q_norm(q_emb + qi_emb) 
            
            # history node embedding
            h_emb, hi_emb = self.h_nem(
                hist_feat,
                img_feat,
                hist_feat_mask,
                img_feat_mask        
            )

            h_emb = self.h_attflat_lang(
                h_emb,
                hist_feat_mask
            )

            hi_emb = self.h_attflat_img(
                hi_emb,
                img_feat_mask
            )

            h_emb = self.h_norm(h_emb + hi_emb)
            h_emb = h_emb.unsqueeze(1)
            q_emb = q_emb.unsqueeze(1)             

            # stack each history node
            if i == 0: h_embs = h_emb
            else: h_embs = torch.cat((h_embs, h_emb), dim=1)

            # structural inference between question node and history node
            binary, w_att = self.sgm(q_emb, h_embs)

            # zero padding to binary structures for computing structural loss
            # making (n_batch x n_round x n_round) tensor
            b_pad = Variable(torch.zeros(n_batch, 1, n_round-(i+1)).cuda())
            binary = torch.cat((binary, b_pad), dim=2)
            binary_strct.append(binary)

            # zero padding to weighted structures for updating
            # making (n_batch x n_round+1 x n_round+1) tensor
            w_pad = Variable(torch.zeros(n_batch, 1, n_round-i).cuda())
            w_att = torch.cat((w_att, w_pad), dim=2)
            weighted_strct = torch.cat((weighted_strct, w_att), dim=1)

            adj = weighted_strct[:, :, :i+2] 
            z = self.sgm.update(q_emb, h_embs, adj)
            enc_outs.append(z)
            
        binary_strct = torch.cat(binary_strct, dim=1)
        enc_out = torch.cat(enc_outs, dim=1)
        enc_out = self.j_proj(enc_out)
        enc_out = torch.tanh(enc_out)
        return enc_out, binary_strct


class AttFlat(nn.Module):
    def __init__(self, __C):
        super(AttFlat, self).__init__()
        self.__C = __C

        self.mlp = MLP(
            in_size=__C['hidden_size'],
            mid_size=__C['flat_mlp'],
            out_size=__C['flat_glimpses'],
            dropout_r=__C['model_dropout'],
            use_relu=True
        )

        self.linear_merge = nn.Linear(
            __C['hidden_size'] * __C['flat_glimpses'],
            __C['flat_out_size']
        )

    def forward(self, x, x_mask):
        att = self.mlp(x)
        att = att.masked_fill(
            x_mask.squeeze(1).squeeze(1).unsqueeze(2),
            -1e9
        )
        att = F.softmax(att, dim=1)

        att_list = []
        for i in range(self.__C['flat_glimpses']):
            att_list.append(
                torch.sum(att[:, :, i: i + 1] * x, dim=1)
            )

        x_atted = torch.cat(att_list, dim=1)
        x_atted = self.linear_merge(x_atted)

        return x_atted

