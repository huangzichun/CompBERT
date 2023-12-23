import argparse
import pickle
import torch
import sys, os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,3'
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



def train(args=None):

    # BERT inilial
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://')

    from transformers import BertTokenizer, BertModel, BertConfig
    device = torch.device("cuda:" + str(args.local_rank))
    config_path = "../config_path/bert_config.json"
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')#.to(device)
    model = BertModel.from_pretrained("bert-base-uncased"
                                      , config=BertConfig.from_pretrained(config_path)
                                      , force_download=False
                                      ).to(device)
    #torch.cuda.set_device(args.local_rank)
    #torch.distributed.init_process_group(backend='nccl', init_method='env://')

    print("Loading Train Dataset", args.train_dataset)
    train_dataset = BERTDataset(args.train_dataset, tokenizer, seq_len=args.seq_len,
                                corpus_lines=args.corpus_lines, on_memory=args.on_memory)
    # train_dataset.subset(200000)

    print("Loading Test Dataset", args.test_dataset)
    test_dataset = BERTDataset(args.test_dataset, tokenizer, seq_len=args.seq_len, on_memory=args.on_memory) \
        if args.test_dataset is not None else None

    # load sememe dataset
    sememe_idx = pickle.load(open("/home/huangchen/data/sememe.idx", "rb"))
    word_sememe_idx = pickle.load(open("/home/huangchen/data/HowNet.idx", "rb"))
    word_sememe_idx = {tokenizer.convert_tokens_to_ids(tokenizer.tokenize(k))[0]: v for k, v in word_sememe_idx.items() if k in tokenizer.vocab}

    print("Creating Dataloader")
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, sampler=train_sampler)
    test_data_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers) \
        if test_dataset is not None else None

    print("Building BERT model with {}".format("discrete mode" if args.discrete else "continuous mode"))
    bert = BERT(len(tokenizer.vocab), hidden=args.hidden, n_layers=args.layers, attn_heads=args.attn_heads) if not args.discrete \
        else Discrete_BERT(len(tokenizer.vocab), model.embeddings, hidden=args.hidden, n_layers=args.layers, attn_heads=args.attn_heads
                           , discrete_only=args.discrete_only, n_codes=3152)

    # bert.init_by_huggingface(model)

    print("Creating BERT Trainer")
    trainer = BERTTrainer(bert, len(tokenizer.vocab), train_dataloader=train_data_loader, test_dataloader=test_data_loader,
                          lr=args.lr, betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.adam_weight_decay,
                          with_cuda=args.with_cuda, cuda_devices=args.cuda_devices, log_freq=args.log_freq,
                          is_discrete=args.discrete, word_sememe_idx=word_sememe_idx, n_sememe=len(sememe_idx), local_rank=args.local_rank)

    print("Training Start")
    for epoch in range(args.epochs):
        #if args.local_rank == 0:
        #    trainer.save(epoch, args.output_path)
        trainer.train_data.sampler.set_epoch(epoch=epoch)
        trainer.train(epoch)
        if args.local_rank == 0:
            trainer.save(epoch, args.output_path)
    
        if test_data_loader is not None:
            trainer.test(epoch)


if __name__ == "__main__":
    # constants
    corpus_wikitext = "/home/huangchen/data/full_wikitext"
    corpus_book = "/home/huangchen/data/full_booktext"
    corpus_file = "/home/huangchen/data/corpus.small"
    corpus_test_file = "/home/huangchen/data/corpus_test"
    vocab_file = "/home/huangchen/data/vocab.small"
    model_file = "/home/huangchen/model/bert"
    discrete = True  # if use discrete_BERT
    discrete_only = False  # if only use discrete_x, otherwise x + discrete_x
    init_bert_name = "bert-base-uncased"

    #wikipedia = load_dataset("wikipedia", "20220301.en")
    #print("done reading..")
    #wikipedia_dataset = wikipedia['train'].remove_columns(['id', 'url', 'title'])#.select(range(int(0.2 * len(wikipedia['train']))))
    #wikipedia_dataset.to_csv("../data/full_wikitext", num_proc=2, header=0, index=0)
    # #
    #exit(0)
    # data set generation for wikitext-2 or 103, saved into corpus_file
    #clean_wikitext_to_file(corpus_wikitext, corpus_file, p=1.1)
    # dataset = load_dataset("bookcorpus")
    # dataset['train'].to_csv("../data/full_booktext", num_proc=2, header=0, index=0)

    # data set creation
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--corpus_path", type=str, default=corpus_file)
    parser.add_argument("-o", "--output_path", type=str, default=vocab_file)
    parser.add_argument("-s", "--vocab_size", type=int, default=None)
    parser.add_argument("-e", "--encoding", type=str, default="utf-8")
    parser.add_argument("-m", "--min_freq", type=int, default=1)
    parser.add_argument("--local_rank", type=int, default=-1)
    args = parser.parse_args()
    #build(args)

    # bert training
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--discrete", type=bool, default=discrete, help="use CAT or BERT")
    parser.add_argument("-x", "--discrete_only", type=bool, default=discrete_only, help="use discrete or mixed CAT (CompBERT)")
    parser.add_argument("-c", "--train_dataset", type=str, help="train dataset for train bert", default=corpus_file)
    parser.add_argument("-t", "--test_dataset", type=str, default=None, help="test set for evaluate train set")
    parser.add_argument("-v", "--vocab_path", type=str, help="built vocab model path with bert-vocab", default=vocab_file)
    parser.add_argument("-o", "--output_path", type=str, help="ex)output/bert.model", default=model_file)

    parser.add_argument("-hs", "--hidden", type=int, default=768, help="hidden size of transformer model")
    parser.add_argument("-l", "--layers", type=int, default=4, help="number of layers")
    parser.add_argument("-a", "--attn_heads", type=int, default=12, help="number of attention heads")
    parser.add_argument("-s", "--seq_len", type=int, default=128, help="maximum sequence len")

    parser.add_argument("-b", "--batch_size", type=int, default=60, help="number of batch_size")  # max 20 * 8
    parser.add_argument("-e", "--epochs", type=int, default=40, help="number of epochs")
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("-w", "--num_workers", type=int, default=0, help="dataloader worker size")

    parser.add_argument("--with_cuda", type=bool, default=True, help="training with CUDA: true, or false")
    parser.add_argument("--log_freq", type=int, default=500, help="printing loss every n iter: setting n")
    parser.add_argument("--corpus_lines", type=int, default=None, help="total number of lines in corpus")
    parser.add_argument("--cuda_devices", type=int, nargs='+', default=None, help="CUDA device ids")
    parser.add_argument("--on_memory", type=bool, default=True, help="Loading on memory: true or false")

    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate of adam")
    parser.add_argument("--adam_weight_decay", type=float, default=2e-4, help="weight_decay of adam")
    parser.add_argument("--warmup", type=int, default=10000)
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="adam first beta value")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="adam first beta value")
    args = parser.parse_args()

    train(args)
