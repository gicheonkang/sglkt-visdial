"""
Reasoning Visual Dialog with Sparse Graph Learning and Knowledge Transfer
Gi-Cheon Kang, Junseok Park, Hwaran Lee, Byoung-Tak Zhang, Jin-Hwa Kim
https://arxiv.org/abs/2004.06698
"""
import numpy as np
import torch, math
import torch.nn as nn
import torch.nn.functional as F
from .net_utils import MLP
from torch.autograd import Variable
from torch.nn.utils.weight_norm import weight_norm

class SANet(nn.Module):
    def __init__(self, __C):
        super(SANet, self).__init__()
        self.n_head = 8
        self.d_hid = __C['hidden_size']
        self.d_hid_head = __C['hidden_size'] // 8
        self.gs = GumbelSoftmax(d_in=self.d_hid_head, num_cls=2, dropout=__C['model_dropout'])

        self.linear_q = nn.Linear(__C['hidden_size'], __C['hidden_size'])
        self.linear_k = nn.Linear(__C['hidden_size'], __C['hidden_size'])
        self.fc = nn.Linear(__C['hidden_size'], __C['hidden_size'])

    def attention(self, q, k):
        logit = q.unsqueeze(2) * k.unsqueeze(3)
        logit = logit.transpose(2, 3)
        attn = logit.sum(-1) / math.sqrt(self.d_hid_head)

        binary = self.gs(logit)
        attn = attn * binary
        attn = F.normalize(attn, p=2, dim=-1)**2
        return binary, attn

    def forward(self, q, k):
        n_batch = q.size(0)
        q = self.linear_q(q).view(
            n_batch,
            -1, 
            self.n_head, 
            self.d_hid_head
        ).transpose(1, 2)

        k = self.linear_k(k).view(
            n_batch, 
            -1, 
            self.n_head, 
            self.d_hid_head
        ).transpose(1, 2)

        binary, attn = self.attention(q, k)
        binary = binary.mean(dim=1)
        attn = attn.mean(dim=1)
        return binary, attn

class GumbelSoftmax(nn.Module):
    '''
    Softmax Relaxation for Gumbel Max Trick 
    '''
    def __init__(self, d_in, num_cls, dropout):
        super().__init__()
        self.linear_g = MLP(
            in_size=d_in,
            mid_size=d_in//2,
            out_size=num_cls,
            dropout_r=dropout,
            use_relu=True
        ) 

        self.logsoftmax = nn.LogSoftmax(dim=-1)

    def st_gumbel_softmax(self, x, temperature=0.5):
        '''
        Straight Throught Gumbel Softmax
        '''
        eps = 1e-20
        noise = Variable(torch.rand(x.size()).cuda())

        noise.data.add_(eps).log_().neg_()
        noise.data.add_(eps).log_().neg_()

        y = (x + noise) / temperature
        y = F.softmax(y, dim=-1)

        shape = y.size()
        _, ind = y.max(dim=-1)
        y_hard = torch.zeros_like(y).view(-1, shape[-1])
        y_hard.scatter_(1, ind.view(-1, 1), 1)
        y_hard = y_hard.view(*shape)
        y_hard = (y_hard - y).detach() + y
        return y_hard

    def forward(self, rel):
        x = self.linear_g(rel)
        x = self.logsoftmax(x)
        
        if self.training:
            mask = self.st_gumbel_softmax(x)
        else:
            _, ind = x.detach().max(4, keepdim=True)     
            mask = x.detach().clone().zero_().scatter_(4, ind, 1)            
        mask = mask[:, :, :, :, -1]
        return mask
