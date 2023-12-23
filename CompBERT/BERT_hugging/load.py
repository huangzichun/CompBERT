import torch.nn as nn
import torch
import sys, os
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from torch.utils.data import DataLoader
from collections import Counter
from BERT_hugging.model import BERT, Discrete_BERT
from BERT_hugging.trainer import BERTTrainer
from BERT_hugging.dataset import BERTDataset, WordVocab
from BERT_hugging.dataset.vocab import build, Vocab
from BERT_hugging.dataset.WikiTextTools import clean_wikitext_to_file
from datasets import load_dataset

torch.load("/home/huangchen/model/bert.ep0", map_location='cpu')
