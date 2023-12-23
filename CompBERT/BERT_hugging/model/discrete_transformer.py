import torch.nn as nn
import torch

from .attention import MultiHeadedAttention, SparseAttention
from .utils import SublayerConnection, PositionwiseFeedForward


class DiscreteTransformerBlock(nn.Module):
    """
    Bidirectional Encoder = Transformer (self-attention)
    Transformer = MultiHead_Attention + Feed_Forward with sublayer connection
    """

    def __init__(self, symbol_hidden_size, max_symbol, feed_forward_hidden, dropout, attn_heads):
        """
        :param hidden: hidden size of transformer
        :param attn_heads: head sizes of multi-head attention
        :param feed_forward_hidden: feed_forward_hidden, usually 4*hidden_size
        :param dropout: dropout rate
        """

        super().__init__()
        self.attention = MultiHeadedAttention(h=attn_heads, d_model=symbol_hidden_size, sparse=False)  # disable sparse attention
        self.sparse_attention = MultiHeadedAttention(h=1, d_model=symbol_hidden_size, sparse=True, stop_gradient=False)  # TODO  stop_gradient=False or True
        self.symbol_mapping = PositionwiseFeedForward(d_model=symbol_hidden_size, d_ff=feed_forward_hidden, dropout=dropout) # useless
        self.input_sublayer = SublayerConnection(size=symbol_hidden_size, dropout=dropout)
        self.output_sublayer = SublayerConnection(size=symbol_hidden_size, dropout=dropout)
        self.reshape_layer = nn.Sequential(nn.Linear(symbol_hidden_size, max_symbol), nn.Softmax(dim=2))  # useless
        self.feed_forward = PositionwiseFeedForward(d_model=symbol_hidden_size, d_ff=feed_forward_hidden, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)

        self.max_symbol = max_symbol
        self.symbol_hidden_size = symbol_hidden_size
        self.symbols = nn.Parameter(torch.empty(max_symbol, symbol_hidden_size))
        self._reset_network()

    def _reset_network(self):
        nn.init.xavier_normal_(self.symbols, gain=nn.init.calculate_gain('relu'))

    def forward(self, x, mask):
        x = self.input_sublayer(x, lambda _x: self.attention.forward(_x, _x, _x, mask=mask))
        x = self.output_sublayer(x, self.feed_forward)
        x_continuous = self.dropout(x)

        # decompose
        # coef = self.reshape_layer(self.output_sublayer(x, self.symbol_mapping))
        # compose
        # x_discrete = self.symbols(coef)

        # sparse attention based discretion, conditioned on context [CLS]

        # we do not want the discrete to effect the continuous, Thus, stop gradient
        x_discrete, coef = self.sparse_attention(query=x_continuous.detach(), key=self.symbols.repeat(x.shape[0],1,1)
                                                 , value=self.symbols.repeat(x.shape[0],1,1)
                                                 , coef=True)

        return x_continuous, x_discrete, coef


# 自定义梯度计算
class TBGradFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        pass

    @staticmethod
    def backward(ctx, grad_outputs):
        pass