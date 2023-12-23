import argparse
import pickle

from torch.utils.data import DataLoader

from BERT.model import BERT, Discrete_BERT
from BERT.trainer import BERTTrainer
from BERT.dataset import BERTDataset, WordVocab
from BERT.dataset.vocab import build
from BERT.dataset.WikiTextTools import clean_wikitext_to_file


def train(args=None):
    # parser = argparse.ArgumentParser()
    # parser.add_argument("-d", "--discrete_only", type=bool, required=True, help="use discrete or continuous BERT")
    # parser.add_argument("-x", "--discrete", type=bool, required=True, help="use discrete or continuous BERT")
    # parser.add_argument("-c", "--train_dataset", required=True, type=str, help="train dataset for train bert")
    # parser.add_argument("-t", "--test_dataset", type=str, default=None, help="test set for evaluate train set")
    # parser.add_argument("-v", "--vocab_path", required=True, type=str, help="built vocab model path with bert-vocab")
    # parser.add_argument("-o", "--output_path", required=True, type=str, help="ex)output/bert.model")
    #
    # parser.add_argument("-hs", "--hidden", type=int, default=256, help="hidden size of transformer model")
    # parser.add_argument("-l", "--layers", type=int, default=8, help="number of layers")
    # parser.add_argument("-a", "--attn_heads", type=int, default=8, help="number of attention heads")
    # parser.add_argument("-s", "--seq_len", type=int, default=20, help="maximum sequence len")
    #
    # parser.add_argument("-b", "--batch_size", type=int, default=64, help="number of batch_size")
    # parser.add_argument("-e", "--epochs", type=int, default=10, help="number of epochs")
    # parser.add_argument("-w", "--num_workers", type=int, default=5, help="dataloader worker size")
    #
    # parser.add_argument("--with_cuda", type=bool, default=True, help="training with CUDA: true, or false")
    # parser.add_argument("--log_freq", type=int, default=10, help="printing loss every n iter: setting n")
    # parser.add_argument("--corpus_lines", type=int, default=None, help="total number of lines in corpus")
    # parser.add_argument("--cuda_devices", type=int, nargs='+', default=None, help="CUDA device ids")
    # parser.add_argument("--on_memory", type=bool, default=True, help="Loading on memory: true or false")
    #
    # parser.add_argument("--lr", type=float, default=1e-3, help="learning rate of adam")
    # parser.add_argument("--adam_weight_decay", type=float, default=0.01, help="weight_decay of adam")
    # parser.add_argument("--adam_beta1", type=float, default=0.9, help="adam first beta value")
    # parser.add_argument("--adam_beta2", type=float, default=0.999, help="adam first beta value")
    #
    # args = parser.parse_args() if not args else args

    print("Loading Vocab", args.vocab_path)
    vocab = WordVocab.load_vocab(args.vocab_path)
    print("Vocab Size: ", len(vocab))

    print("Loading Train Dataset", args.train_dataset)
    train_dataset = BERTDataset(args.train_dataset, vocab, seq_len=args.seq_len,
                                corpus_lines=args.corpus_lines, on_memory=args.on_memory)

    print("Loading Test Dataset", args.test_dataset)
    test_dataset = BERTDataset(args.test_dataset, vocab, seq_len=args.seq_len, on_memory=args.on_memory) \
        if args.test_dataset is not None else None

    # load sememe dataset
    sememe_idx = pickle.load(open("../data/sememe.idx", "rb"))
    word_sememe_idx = pickle.load(open("../data/HowNet.idx", "rb"))
    word_sememe_idx = {vocab.stoi.get(k):v for k, v in word_sememe_idx.items() if k in vocab.stoi}

    print("Creating Dataloader")
    train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers)
    test_data_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers) \
        if test_dataset is not None else None

    print("Building BERT model with {}".format("discrete mode" if args.discrete else "continuous mode"))
    bert = BERT(len(vocab), hidden=args.hidden, n_layers=args.layers, attn_heads=args.attn_heads) if not args.discrete \
        else Discrete_BERT(len(vocab), hidden=args.hidden, n_layers=args.layers, attn_heads=args.attn_heads
                           , discrete_only=args.discrete_only, n_codes=3152)

    print("Creating BERT Trainer")
    trainer = BERTTrainer(bert, len(vocab), train_dataloader=train_data_loader, test_dataloader=test_data_loader,
                          lr=args.lr, betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.adam_weight_decay,
                          with_cuda=args.with_cuda, cuda_devices=args.cuda_devices, log_freq=args.log_freq,
                          is_discrete=args.discrete, word_sememe_idx=word_sememe_idx, n_sememe=len(sememe_idx))

    print("Training Start")
    for epoch in range(args.epochs):
        trainer.train(epoch)
        trainer.save(epoch, args.output_path)

        if test_data_loader is not None:
            trainer.test(epoch)


if __name__ == "__main__":
    # constants
    corpus_wikitext = "../data/wikitext"
    corpus_file = "../data/corpus.small"
    vocab_file = "../data/vocab.small"
    model_file = "../model/bert"
    discrete = True  # if use discrete_BERT
    discrete_only = False  # if only use discrete_x, otherwise x + discrete_x

    # data set generation for wikitext-2 or 103, saved into corpus_file
    clean_wikitext_to_file(corpus_wikitext, corpus_file, p=1.1)

    # data set creation
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--corpus_path", type=str, default=corpus_file)
    parser.add_argument("-o", "--output_path", type=str, default=vocab_file)
    parser.add_argument("-s", "--vocab_size", type=int, default=None)
    parser.add_argument("-e", "--encoding", type=str, default="utf-8")
    parser.add_argument("-m", "--min_freq", type=int, default=1)
    args = parser.parse_args()
    build(args)

    # bert training
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--discrete", type=bool, default=discrete, help="use discrete or continuous BERT")
    parser.add_argument("-x", "--discrete_only", type=bool, default=discrete_only, help="use discrete or continuous BERT")
    parser.add_argument("-c", "--train_dataset", type=str, help="train dataset for train bert", default=corpus_file)
    parser.add_argument("-t", "--test_dataset", type=str, default=None, help="test set for evaluate train set")
    parser.add_argument("-v", "--vocab_path", type=str, help="built vocab model path with bert-vocab", default=vocab_file)
    parser.add_argument("-o", "--output_path", type=str, help="ex)output/bert.model", default=model_file)

    parser.add_argument("-hs", "--hidden", type=int, default=256, help="hidden size of transformer model")
    parser.add_argument("-l", "--layers", type=int, default=8, help="number of layers")
    parser.add_argument("-a", "--attn_heads", type=int, default=8, help="number of attention heads")
    parser.add_argument("-s", "--seq_len", type=int, default=20, help="maximum sequence len")

    parser.add_argument("-b", "--batch_size", type=int, default=64, help="number of batch_size")
    parser.add_argument("-e", "--epochs", type=int, default=10, help="number of epochs")
    parser.add_argument("-w", "--num_workers", type=int, default=1, help="dataloader worker size")

    parser.add_argument("--with_cuda", type=bool, default=True, help="training with CUDA: true, or false")
    parser.add_argument("--log_freq", type=int, default=10, help="printing loss every n iter: setting n")
    parser.add_argument("--corpus_lines", type=int, default=None, help="total number of lines in corpus")
    parser.add_argument("--cuda_devices", type=int, nargs='+', default=None, help="CUDA device ids")
    parser.add_argument("--on_memory", type=bool, default=True, help="Loading on memory: true or false")

    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate of adam")
    parser.add_argument("--adam_weight_decay", type=float, default=0.01, help="weight_decay of adam")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="adam first beta value")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="adam first beta value")
    args = parser.parse_args()
    train(args)
