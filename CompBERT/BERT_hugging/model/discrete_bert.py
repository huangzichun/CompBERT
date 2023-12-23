import torch
import torch.nn as nn

from .discrete_transformer import DiscreteTransformerBlock
from .embedding import BERTEmbedding
from .utils import SublayerConnection
from .attention import MultiHeadedAttention


class Discrete_BERT(nn.Module):
    """
    BERT model : Bidirectional Encoder Representations from Transformers.
    """

    def __init__(self, vocab_size, bert_embedding, hidden=768, n_layers=12, attn_heads=12, dropout=0.1, n_codes=1000, discrete_only=False):
        """
        :param vocab_size: vocab_size of total words
        :param hidden: BERT model hidden size
        :param n_layers: numbers of Transformer blocks(layers)
        :param attn_heads: number of attention heads
        :param dropout: dropout rate
        """

        super().__init__()
        self.hidden = hidden
        self.n_layers = n_layers
        self.attn_heads = attn_heads
        self.n_codes = n_codes

        # paper noted they used 4*hidden_size for ff_network_hidden_size
        self.feed_forward_hidden = hidden * 4

        # embedding for BERT, sum of positional, segment, token embeddings
        # self.embedding = BERTEmbedding(vocab_size=vocab_size, embed_size=hidden)
        self.bert_embedding = bert_embedding

        # multi-layers transformer blocks, deep network
        self.transformer_blocks = nn.ModuleList(
            [DiscreteTransformerBlock(symbol_hidden_size=hidden, max_symbol=hidden, feed_forward_hidden=hidden * 4
                                      , dropout=dropout, attn_heads=self.attn_heads) for _ in range(n_layers - 1)])

        sememe_layer = DiscreteTransformerBlock(symbol_hidden_size=hidden, max_symbol=self.n_codes, feed_forward_hidden=hidden * 4
                                                , dropout=dropout, attn_heads=self.attn_heads)
        self.transformer_blocks.append(sememe_layer)

        # adjusting continuous and discrete
        self.adjust_layer = MultiHeadedAttention(h=1, d_model=hidden, sparse=False)# nn.Sequential(nn.Linear(hidden, 2), nn.Softmax(dim=2))

        self.discrete_only = discrete_only

    def forward(self, x, segment_info, mask=None):
        # attention masking for padded token
        # torch.ByteTensor([batch_size, 1, seq_len, seq_len)
        mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1) if mask is None else mask.unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)

        # embedding the indexed sequence to sequence of vectors
        # x = self.embedding(x, segment_info)
        # print(segment_info)
        x = self.bert_embedding(input_ids=x, token_type_ids=segment_info)

        # running over multiple transformer blocks
        coefs = []
        x_gap = []
        x_discrete = x
        x_continue = x
        # TODO: 是否需要同时输出离散，连续和混合表达。gap是否需要每一层都保留和约束
        for transformer in self.transformer_blocks:
            x_continue, x_discrete, coef = transformer.forward(x, mask)
            # weight = self.adjust_layer(x)
            if self.discrete_only:
                x = x_discrete
            else:
                # x = (x + x_discrete) / 2.0
                # x = weight[:, :, 0].unsqueeze(2) * x + (1.0 - weight[:, :, 1]).unsqueeze(2) * x
                x = torch.cat((x_continue, x_discrete), dim=1)
                x = self.adjust_layer(x, x, x)  # self attention based merge, TODO shall we sparse
                half = int(x.shape[1]/2)
                x = (x[:, :half, :] + x[:, half:, :]) / 2.0  # TODO, shall we use x[:, :100, :] or x[:, 100:, :], without addition
            coefs.append(coef)
            # if x_gap is minimized, it should focus on the discrete part. Thus, we stop the gradient of continue part
            x_gap.append(x_continue - x_discrete)
        return x, x_discrete, x_continue, x_gap, coefs


    def init_by_huggingface(self, pre_model):
        print("initialization begin")
        for block_ind in range(len(self.transformer_blocks)):
            self.init_by_huggingface_(self.transformer_blocks[block_ind], pre_model.encoder.layer[block_ind])

    def init_by_huggingface_(self, my_transformer, his_transformer):
        print("initialization for transformer begin")
        # multi-attention query, key, value
        my_transformer.attention.linear_layers[0].load_state_dict(his_transformer.attention.self.query.state_dict())
        my_transformer.attention.linear_layers[1].load_state_dict(his_transformer.attention.self.key.state_dict())
        my_transformer.attention.linear_layers[2].load_state_dict(his_transformer.attention.self.value.state_dict())
        my_transformer.attention.output_linear.load_state_dict(his_transformer.attention.output.dense.state_dict())

        my_transformer.input_sublayer.norm.a_2 = his_transformer.attention.output.LayerNorm.weight
        my_transformer.input_sublayer.norm.b_2 = his_transformer.attention.output.LayerNorm.bias
        my_transformer.input_sublayer.norm.eps = his_transformer.attention.output.LayerNorm.eps

        my_transformer.feed_forward.w_1.load_state_dict(his_transformer.intermediate.dense.state_dict())
        my_transformer.feed_forward.w_2.load_state_dict(his_transformer.output.dense.state_dict())

        my_transformer.output_sublayer.norm.a_2 = his_transformer.output.LayerNorm.weight
        my_transformer.output_sublayer.norm.b_2 = his_transformer.output.LayerNorm.bias
        my_transformer.output_sublayer.norm.eps = his_transformer.output.LayerNorm.eps
