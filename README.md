# CompBERT
Code of [Towards Equipping Transformer with the Ability of Systematic Compositionality (AAAI 2024)](https://arxiv.org/abs/2312.07280).

# Q & A
1. This code is based on [Pytorch-BERT](https://github.com/codertimo/BERT-pytorch)
2. In the `BERT_hugging` folder, we have added the following files to support training on discrete variables.
	- model/discrete_bert.py
	- model/discrete_transformer.py
3. In the `BERT_hugging` folder, we have modified the following files.
	- The contents in the attention folder have been updated to include sparse attention and multi-head attention based on sparse attention.
	- The trainer/pretrain.py file has been modified to include two additional pre-training tasks.
4. The main_bert.py in `BERT_hugging` now has two new parameters: "discrete=True" for training CompBERT, otherwise it trains traditional BERT; "discrete_only=True" for using a mixture of discrete and continuous representations, otherwise it only uses discrete representations. The "discrete_only" parameter is inactive when "discrete=False".
5. The `data` folder contains pre-training data, consistent with BERT, in the same format as in BERT-pytorch. Please download the data from the internet and clean them. We have added openhownet data to supervise our two additional pre-training tasks. We provide the processed files sememe.idx and HowNet.idx.

# Reference
If you make advantage of the CompBERT in your research, please cite the following in your manuscript:

```
@inproceedings{compbert2024,
  title={Towards Equipping Transformer with the Ability of Systematic Compositionality},
  author={Chen Huang, Peixin Qin, Wenqiang Lei, Jiancheng Lv},
  booktitle = {{AAAI}},
  publisher = {{AAAI} Press},
  year      = {2024}
}
```