"""
Reasoning Visual Dialog with Sparse Graph Learning and Knowledge Transfer
Gi-Cheon Kang, Junseok Park, Hwaran Lee, Byoung-Tak Zhang, Jin-Hwa Kim
https://arxiv.org/abs/2004.06698
"""
import torch, math
import torch.nn.functional as F

from torch import nn
from .sparse import SANet
from .net_utils import FC, MLP, LayerNorm

class NodeEmbeddingModule(nn.Module):
    def __init__(self, __C):
        super(NodeEmbeddingModule, self).__init__()
        self.enc_list = nn.ModuleList([SA(__C) for _ in range(__C['transformer_num_layers'])])
        self.dec_list = nn.ModuleList([SGA(__C) for _ in range(__C['transformer_num_layers'])])

    def forward(self, x, y, x_mask, y_mask):
        # Get hidden vector
        for enc in self.enc_list:
            x = enc(x, x_mask)

        for dec in self.dec_list:
            y = dec(y, x, y_mask, x_mask)

        return x, y

class SparseGraphLearningModule(nn.Module):
    def __init__(self, __C):
        super(SparseGraphLearningModule, self).__init__()
        self.sparse = SANet(__C)
        self.upm = nn.ModuleList([UP(__C) for _ in range(__C['sgl_update_num_layers'])])

    def forward(self, x, y):
        binary, w_att = self.sparse(x, y)
        return binary, w_att

    def update(self, x, y, adj):
        x = torch.cat((y, x), dim=1)
        for up in self.upm:
            x = up(x, adj)

        return x[:, -1:, :]


class UP(nn.Module):
    def __init__(self, __C):
        super(UP, self).__init__() 
        self.dropout1 = nn.Dropout(__C['model_dropout'])
        self.dropout2 = nn.Dropout(__C['model_dropout'])
        self.norm1 = LayerNorm(__C['hidden_size'])
        self.norm2 = LayerNorm(__C['hidden_size'])
        self.ffn = FFN(__C)

    def forward(self, x, adj): 
        output = torch.matmul(adj, x)
        x = self.norm1(x + self.dropout1(output))
        x = self.norm2(x + self.dropout2(self.ffn(x)))
        return x


"""
The code below is from MCAN-VQA, https://github.com/MILVLG/mcan-vqa
"""
# ------------------------------
# ---- Multi-Head Attention ----
# ------------------------------

class MHAtt(nn.Module):
    def __init__(self, __C):
        super(MHAtt, self).__init__()
        self.__C = __C

        self.linear_v = nn.Linear(__C['hidden_size'], __C['hidden_size'])
        self.linear_k = nn.Linear(__C['hidden_size'], __C['hidden_size'])
        self.linear_q = nn.Linear(__C['hidden_size'], __C['hidden_size'])
        self.linear_merge = nn.Linear(__C['hidden_size'], __C['hidden_size'])

        self.dropout = nn.Dropout(__C['model_dropout'])

    def forward(self, v, k, q, mask):
        n_batches = q.size(0)

        v = self.linear_v(v).view(
            n_batches,
            -1,
            self.__C['multi_head'],
            self.__C['hidden_size_head']
        ).transpose(1, 2)

        k = self.linear_k(k).view(
            n_batches,
            -1,
            self.__C['multi_head'],
            self.__C['hidden_size_head']
        ).transpose(1, 2)

        q = self.linear_q(q).view(
            n_batches,
            -1,
            self.__C['multi_head'],
            self.__C['hidden_size_head']
        ).transpose(1, 2)

        atted = self.att(v, k, q, mask)
        atted = atted.transpose(1, 2).contiguous().view(
            n_batches,
            -1,
            self.__C['hidden_size']
        )

        atted = self.linear_merge(atted)

        return atted

    def att(self, value, key, query, mask):
        d_k = query.size(-1)

        scores = torch.matmul(
            query, key.transpose(-2, -1)
        ) / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)

        att_map = F.softmax(scores, dim=-1)
        att_map = self.dropout(att_map)

        return torch.matmul(att_map, value)


# ---------------------------
# ---- Feed Forward Nets ----
# ---------------------------

class FFN(nn.Module):
    def __init__(self, __C):
        super(FFN, self).__init__()

        self.mlp = MLP(
            in_size=__C['hidden_size'],
            mid_size=__C['hidden_size'] * 4,
            out_size=__C['hidden_size'],
            dropout_r=__C['model_dropout'],
            use_relu=True
        )

    def forward(self, x):
        return self.mlp(x)


# ------------------------
# ---- Self Attention ----
# ------------------------

class SA(nn.Module):
    def __init__(self, __C):
        super(SA, self).__init__()

        self.mhatt = MHAtt(__C)
        self.ffn = FFN(__C)

        self.dropout1 = nn.Dropout(__C['model_dropout'])
        self.norm1 = LayerNorm(__C['hidden_size'])

        self.dropout2 = nn.Dropout(__C['model_dropout'])
        self.norm2 = LayerNorm(__C['hidden_size'])

    def forward(self, x, x_mask):
        x = self.norm1(x + self.dropout1(
            self.mhatt(x, x, x, x_mask)
        ))

        x = self.norm2(x + self.dropout2(
            self.ffn(x)
        ))

        return x


# -------------------------------
# ---- Self Guided Attention ----
# -------------------------------

class SGA(nn.Module):
    def __init__(self, __C):
        super(SGA, self).__init__()

        self.mhatt1 = MHAtt(__C)
        self.mhatt2 = MHAtt(__C)
        self.ffn = FFN(__C)

        self.dropout1 = nn.Dropout(__C['model_dropout'])
        self.norm1 = LayerNorm(__C['hidden_size'])

        self.dropout2 = nn.Dropout(__C['model_dropout'])
        self.norm2 = LayerNorm(__C['hidden_size'])

        self.dropout3 = nn.Dropout(__C['model_dropout'])
        self.norm3 = LayerNorm(__C['hidden_size'])

    def forward(self, x, y, x_mask, y_mask):
        x = self.norm1(x + self.dropout1(
            self.mhatt1(x, x, x, x_mask)
        ))

        x = self.norm2(x + self.dropout2(
            self.mhatt2(y, y, x, y_mask)
        ))

        x = self.norm3(x + self.dropout3(
            self.ffn(x)
        ))

        return x

