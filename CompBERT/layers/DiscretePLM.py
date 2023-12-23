import torch
import torch.nn as nn
from DecomposeLayer import DecomposeLayer


class DiscretePLM(nn.Module):
    def __init__(self, max_symbol1, max_symbol2, max_symbol3, symbol_hidden_size):
        super(DiscretePLM, self).__init__()
        self.one_layer = DecomposeLayer(max_symbol1, symbol_hidden_size)
        self.two_layer = DecomposeLayer(max_symbol2, symbol_hidden_size)
        self.three_layer = DecomposeLayer(max_symbol3, symbol_hidden_size)

    def forward(self, x, mixup=False):
        x_hat1, x_gap1, coef1 = self.one_layer(x)
        x_hat2, x_gap2, coef2 = self.one_layer(x_hat1)
        x_hat3, x_gap3, coef3 = self.one_layer(x_hat2)

        if not mixup:
            return torch.hstack((x_hat1, x_hat2, x_hat3))\
                , torch.hstack((x_gap1, x_gap2, x_gap3))\
                , torch.hstack((coef1, coef2, coef3))
        else:
            return torch.hstack((x_hat1 + x, x_hat2 + x, x_hat3 + x))\
                , torch.hstack((x_gap1, x_gap2, x_gap3))\
                , torch.hstack((coef1, coef2, coef3))

    def init_weights(self, symbols_1, symbols_2, symbols_3):
        assert isinstance(symbols_1, torch.FloatTensor) and symbols_1.shape == self.one_layer.shape, "one_layer initalization error"
        assert isinstance(symbols_2, torch.FloatTensor) and symbols_2.shape == self.two_layer.shape, "two_layer initalization error"
        assert isinstance(symbols_3, torch.FloatTensor) and symbols_3.shape == self.three_layer.shape, "three_layer initalization error"
        self.one_layer.init_weights(symbols_1)
        self.two_layer.init_weights(symbols_2)
        self.three_layer.init_weights(symbols_3)

