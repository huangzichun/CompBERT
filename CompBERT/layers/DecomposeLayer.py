import torch.nn as nn
from setting.setting import *


class DecomposeLayer(nn.Module):
    def __init__(self, symbol_hidden_size, max_symbol, feed_forward_hidden, dropout):
        super(DecomposeLayer, self).__init__()
        self.max_symbol = max_symbol
        self.symbol_hidden_size = symbol_hidden_size
        self.symbols = nn.Linear(max_symbol, symbol_hidden_size)
        self._reset_network()
        self.symbol_mapping = nn.Sequential(
            nn.Linear(symbol_hidden_size, feed_forward_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feed_forward_hidden, max_symbol),
            nn.Sigmoid()
        )

    def _reset_network(self):
        nn.init.xavier_normal_(self.symbols.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(self.symbols.bias, gain=nn.init.calculate_gain('relu'))

    def forward(self, x):
        # decompose
        coef = self.symbol_mapping(x)

        # compose
        x_hat = self.symbols(coef)

        # residual
        x_gap = x - x_hat
        return x_hat, x_gap, coef
    
    def init_weights(self, symbols_):
        assert isinstance(symbols_, torch.FloatTensor) and symbols_.shape == self.symbols.shape, "initalization error"
        self.symbols.weight.data = symbols_
