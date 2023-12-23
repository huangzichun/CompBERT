import torch.nn as nn

from .bert import BERT


class BERTLM(nn.Module):
    """
    BERT Language Model
    Next Sentence Prediction Model + Masked Language Model
    """

    def __init__(self, bert: BERT, vocab_size):
        """
        :param bert: BERT model which should be trained
        :param vocab_size: total vocab size for masked_lm
        """

        super().__init__()
        self.bert = bert
        self.next_sentence = NextSentencePrediction(self.bert.hidden)
        self.mask_lm = MaskedLanguageModel(self.bert.hidden, vocab_size)

    def forward(self, x, segment_label):
        if isinstance(self.bert, BERT):
            x = self.bert(x, segment_label)
            return self.next_sentence(x), self.mask_lm(x)
        else:
            #TODO: what should i use in the downstreaming task?
            x, x_discrete, x_continuous, x_gaps, coefs = self.bert(x, segment_label)
            return self.next_sentence(x), self.mask_lm(x), self.next_sentence(x_continuous), self.mask_lm(x_continuous), \
                   self.next_sentence(x_discrete), self.mask_lm(x_discrete), \
                   self.next_sentence(x_gaps[-1]), self.mask_lm(x_gaps[-1]), \
                   x_gaps, x_continuous[-1], x_discrete[-1], coefs


class NextSentencePrediction(nn.Module):
    """
    2-class classification model : is_next, is_not_next
    """

    def __init__(self, hidden):
        """
        :param hidden: BERT model output size
        """
        super().__init__()
        self.linear = nn.Linear(hidden, 2)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        return self.softmax(self.linear(x[:, 0]) + 1e-14)  # add by hc for overflow


class MaskedLanguageModel(nn.Module):
    """
    predicting origin token from masked input sequence
    n-class classification problem, n-class = vocab_size
    """

    def __init__(self, hidden, vocab_size):
        """
        :param hidden: output size of BERT model
        :param vocab_size: total vocab size
        """
        super().__init__()
        self.linear = nn.Linear(hidden, vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        return self.softmax(self.linear(x))
