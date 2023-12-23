import torch.nn as nn
import torch.nn.functional as F
import torch

import math

from BERT.model.utils.layer_norm import RMSNorm


class Attention(nn.Module):
    """
    Compute 'Scaled Dot Product Attention
    """

    def forward(self, query, key, value, mask=None, dropout=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(query.size(-1))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        p_attn = F.softmax(scores, dim=-1)

        if dropout is not None:
            p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn


# Sparse Attention with Linear Units, 2021, EMNLP
class SparseAttention(nn.Module):
    def __init__(self, p, stop_gradient=False):
        super(SparseAttention, self).__init__()
        self.rms_norm = RMSNorm(p)
        self.stop_gradient = stop_gradient  # for optimizing the coefs and directories

    def forward(self, query, key, value, mask=None, dropout=None):
        scores = (torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))) if not self.stop_gradient \
            else (torch.matmul(query, key.transpose(-2, -1).detach()) / math.sqrt(query.size(-1)))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # change softmax to relu
        p_attn = F.relu(scores)

        if dropout is not None:
            p_attn = dropout(p_attn)

        # Stabilization with Normalization
        p_attn = self.rms_norm(p_attn)
        return (torch.matmul(p_attn, value), p_attn) if not self.stop_gradient \
            else (torch.matmul(p_attn.detach(), value), p_attn)
