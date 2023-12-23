import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from ..model import BERTLM, BERT
from ..model.utils.EntropyLoss import EntropyLoss
from .optim_schedule import ScheduledOptim
from functools import reduce
import copy
import tqdm
import pickle
import torch.nn.functional as F


class BERTTrainer:
    """
    BERTTrainer make the pretrained BERT model with two LM training method.

        1. Masked Language Model : 3.3.1 Task #1: Masked LM
        2. Next Sentence prediction : 3.3.2 Task #2: Next Sentence Prediction

    please check the details on README.md with simple example.

    """

    def __init__(self, bert: BERT, vocab_size: int,
                 train_dataloader: DataLoader, test_dataloader: DataLoader = None,
                 lr: float = 1e-4, betas=(0.9, 0.999), weight_decay: float = 0.01, warmup_steps=10000,
                 with_cuda: bool = True, cuda_devices=None, log_freq: int = 10, is_discrete=True
                 , word_sememe_idx=None, sememe_layer=-1, n_sememe=None, local_rank=None):
        """
        :param bert: BERT model which you want to train
        :param vocab_size: total word vocab size
        :param train_dataloader: train dataset data loader
        :param test_dataloader: test dataset data loader [can be None]
        :param lr: learning rate of optimizer
        :param betas: Adam optimizer betas
        :param weight_decay: Adam optimizer weight decay param
        :param with_cuda: traning with cuda
        :param log_freq: logging frequency of the batch iteration
        """

        # Setup cuda device for BERT training, argument -c, --cuda should be true
        cuda_condition = torch.cuda.is_available() and with_cuda
        #self.device = torch.device("cuda:0" if cuda_condition else "cpu")
        self.device = torch.device("cuda:" + str(local_rank) if cuda_condition else "cpu")

        # This BERT model will be saved every epoch
        self.bert = bert
        self.is_discrete = is_discrete
        # Initialize the BERT Language Model, with BERT model
        self.model = BERTLM(bert, vocab_size).to(self.device)

        # Distributed GPU training if CUDA can detect more than 1 GPU
        if with_cuda and torch.cuda.device_count() > 1:
            print("Using %d GPUS for BERT" % torch.cuda.device_count())
            self.model = nn.parallel.DistributedDataParallel(self.model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
            # self.model = nn.parallel.DistributedDataParallel(self.model, device_ids=[local_rank], output_device=local_rank)

        # Setting the train and test data loader
        self.train_data = train_dataloader
        self.test_data = test_dataloader

        # Setting the Adam optimizer with hyper-param
        self.optim = Adam(self.model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
        self.optim_schedule = ScheduledOptim(self.optim, self.bert.hidden, n_warmup_steps=warmup_steps)

        # Using Negative Log Likelihood Loss function for predicting the masked_token
        self.criterion = nn.NLLLoss(ignore_index=0)
        self.criterion_all = nn.NLLLoss()
        self.entro = EntropyLoss()
        self.l1_loss = torch.nn.L1Loss()

        self.log_freq = log_freq

        # sememe info
        self.word_sememe_idx = word_sememe_idx
        self.n_sememe = n_sememe
        self.sememe_layer = sememe_layer  # top layer: 0, or bottom layer: -1

        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))

    def train(self, epoch):
        self.iteration(epoch, self.train_data)

    def test(self, epoch):
        self.iteration(epoch, self.test_data, train=False)

    def iteration(self, epoch, data_loader, train=True):
        """
        loop over the data_loader for training or testing
        if on train status, backward operation is activated
        and also auto save the model every peoch

        :param epoch: current epoch index
        :param data_loader: torch.utils.data.DataLoader for iteration
        :param train: boolean value of is train or test
        :return: None
        """
        str_code = "train" if train else "test"

        # Setting the tqdm progress bar
        data_iter = tqdm.tqdm(enumerate(data_loader),
                              desc="EP_%s:%d" % (str_code, epoch),
                              total=len(data_loader),
                              bar_format="{l_bar}{r_bar}")

        avg_loss = 0.0
        avg_loss_next = 0.0
        avg_loss_mask = 0.0
        avg_loss_rc = 0.0
        avg_loss_next_discrete = 0.0
        avg_loss_mask_discrete = 0.0
        avg_loss_next_continuous = 0.0
        avg_loss_mask_continuous = 0.0
        avg_loss_next_gap = 0.0
        avg_loss_mask_gap = 0.0
        avg_loss_entropy = 0.0
        avg_loss_sparse = 0.0
        avg_loss_sememe = 0.0


        total_correct = 0
        total_correct_discrete = 0
        total_correct_continuous = 0
        total_correct_gap = 0
        total_correct_mask = 0
        total_correct_mask_discrete = 0
        total_correct_mask_continuous = 0
        total_correct_mask_gap = 0
        total_element = 0
        total_element_mask = 0

        for i, data in data_iter:
            # 0. batch_data will be sent into the device(GPU or cpu)
            data = copy.deepcopy(data)
            data = {key: value.to(self.device) for key, value in data.items()}

            # 1. forward the next_sentence_prediction and masked_lm model
            if not self.is_discrete:
                next_sent_output, mask_lm_output = self.model.forward(data["bert_input"], data["segment_label"])
            else:
                next_sent_output, mask_lm_output, next_sent_output_continuous, mask_lm_output_continuous, \
                next_sent_output_discrete, mask_lm_output_discrete, \
                next_sent_output_gap, mask_lm_output_gap, gaps, last_x_continuous, last_x_discrete, coefs = \
                    self.model.forward(data["bert_input"], data["segment_label"])
                last_x_gap = gaps[-1]

            # 2-1. NLL(negative log likelihood) loss of is_next classification result
            next_loss = self.criterion_all(next_sent_output, data["is_next"]) #* torch.mean((data["bert_label"] != 0).sum(1).float())

            # 2-2. NLLLoss of predicting masked token word
            mask_loss = self.criterion(mask_lm_output.transpose(1, 2), data["bert_label"]) #/ mask_lm_output.shape[1]

            # 2-3. Adding next_loss and mask_loss : 3.4 Pre-training Procedure
            loss = next_loss + mask_loss

            # 2-4. Adding discrete part loss
            if self.is_discrete:
                #with torch.no_grad():

                # 2-4-1 RC: reconstruction consistency TODO full layer
                rc_loss = 0.1 * F.mse_loss(last_x_discrete, last_x_continuous.detach()) + F.mse_loss(last_x_discrete.detach(), last_x_continuous)

                # 2-4-2 SC: Semantic Consistency
                # task loss for continuous should be small
                next_loss_continuous = self.criterion_all(next_sent_output_continuous, data["is_next"])
                mask_loss_continuous = self.criterion(mask_lm_output_continuous.transpose(1, 2), data["bert_label"])

                # task loss for discrete should be small
                next_loss_discrete = self.criterion_all(next_sent_output_discrete, data["is_next"])
                mask_loss_discrete = self.criterion(mask_lm_output_discrete.transpose(1, 2), data["bert_label"])

                # 2-4-3 NM: Nuance minimization
                # task loss for gap should be large, task-unrelated
                next_loss_gap = torch.exp(-1.0 * self.criterion_all(next_sent_output_gap, data["is_next"]))
                mask_loss_gap = torch.exp(-1.0 * self.criterion(mask_lm_output_gap.transpose(1, 2), data["bert_label"]))

                # informativeless TODO shall we use the outputs of all layers
                entropy_loss = torch.stack([torch.exp(-1.0 * self.entro(torch.softmax(i, 2))) for i in gaps]).mean()

                # 2-4-4 sparsity loss
                # sparsity loss for coef should be small
                sparsity_loss = torch.mean(torch.FloatTensor([self.l1_loss(coef, torch.zeros(coef.shape).cuda()) for coef in coefs]))

                # 2-4-5 guided decompositional loss TODO: which layer to use sememe? top or bottom
                assert coefs[self.sememe_layer].shape[-1] >= self.n_sememe
                sememes = data['bert_input'].unsqueeze(2).repeat(1, 1, coefs[self.sememe_layer].shape[-1])
                ordered_idx = data['bert_input'].view(-1, 1).squeeze(1).cpu().numpy()
                ordered_sememe = [self.word_sememe_idx[x] if x in self.word_sememe_idx else [] for x in ordered_idx]
                ones = torch.ones(sememes.shape)
                for kk in range(len(ordered_sememe)):
                    ones[int(kk / data['bert_input'].shape[1]), kk % data['bert_input'].shape[1], torch.LongTensor(ordered_sememe[kk])] = 0
                # remove those words uncovered by sememe, we only calculate the loss for wrong sememe-matching
                valid = torch.BoolTensor([1 if len(x) >0 else 0 for x in ordered_sememe]).view(sememes.shape[0], -1).cuda()
                sememe_loss = torch.abs(coefs[self.sememe_layer].squeeze(1) * ones.cuda()).sum(2)
                sememe_loss = (sememe_loss * valid).mean()

                # ALL LOSS
                RC_LOSS = rc_loss
                SC_LOSS = next_loss_continuous + mask_loss_continuous + next_loss_discrete + mask_loss_discrete
                NM_LOSS = entropy_loss + next_loss_gap + mask_loss_gap
                SPARSE_LOSS = sparsity_loss
                DECOMPOSITION_LOSS = sememe_loss
                loss += RC_LOSS + SC_LOSS + NM_LOSS + DECOMPOSITION_LOSS + SPARSE_LOSS
                # loss += SC_LOSS
            else:
                rc_loss = 0.0
                next_loss_discrete = entropy_loss = sparsity_loss = sememe_loss = 0.0
                mask_loss_discrete = mask_loss_continuous = next_loss_gap = mask_loss_gap = next_loss_continuous = 0.0

            # 3. backward and optimization only in train
            if train:
                self.optim_schedule.zero_grad()
                loss.backward()
                self.optim_schedule.step_and_update_lr()

            # next sentence prediction accuracy
            correct = next_sent_output.argmax(dim=-1).eq(data["is_next"]).sum().item()
            correct_continuous = next_sent_output_continuous.argmax(dim=-1).eq(data["is_next"]).sum().item() if "next_sent_output_continuous" in vars() else 0
            correct_discrete = next_sent_output_discrete.argmax(dim=-1).eq(data["is_next"]).sum().item() if "next_sent_output_discrete" in vars() else 0
            correct_gap = next_sent_output_gap.argmax(dim=-1).eq(data["is_next"]).sum().item() if "next_sent_output_gap" in vars() else 0

            # mask acc
            correct_mask = ((mask_lm_output.argmax(dim=-1).eq(data["bert_label"])) * (data["bert_label"] != 0)).sum().item()
            correct_mask_discrete = ((mask_lm_output_discrete.argmax(dim=-1).eq(data["bert_label"])) * (data["bert_label"] != 0)).sum().item() if "mask_lm_output_discrete" in vars() else 0
            correct_mask_continuous = ((mask_lm_output_continuous.argmax(dim=-1).eq(data["bert_label"])) * (data["bert_label"] != 0)).sum().item() if "mask_lm_output_continuous" in vars() else 0
            correct_mask_gap = ((mask_lm_output_gap.argmax(dim=-1).eq(data["bert_label"])) * (data["bert_label"] != 0)).sum().item() if "mask_lm_output_gap" in vars() else 0

            # losses
            avg_loss += loss.item()
            avg_loss_next += next_loss.item()
            avg_loss_mask += mask_loss.item()
            avg_loss_rc += rc_loss.item() if isinstance(rc_loss, torch.Tensor) else rc_loss
            avg_loss_next_discrete += next_loss_discrete.item() if isinstance(next_loss_discrete, torch.Tensor) else next_loss_discrete
            avg_loss_mask_discrete += mask_loss_discrete.item() if isinstance(mask_loss_discrete, torch.Tensor) else mask_loss_discrete
            avg_loss_next_continuous += next_loss_continuous.item() if isinstance(next_loss_continuous, torch.Tensor) else next_loss_continuous
            avg_loss_mask_continuous += mask_loss_continuous.item() if isinstance(mask_loss_continuous, torch.Tensor) else mask_loss_continuous
            avg_loss_next_gap += next_loss_gap.item() if isinstance(next_loss_gap, torch.Tensor) else next_loss_gap
            avg_loss_mask_gap += mask_loss_gap.item() if isinstance(mask_loss_gap, torch.Tensor) else mask_loss_gap
            avg_loss_entropy += entropy_loss.item() if isinstance(entropy_loss, torch.Tensor) else entropy_loss
            avg_loss_sparse += sparsity_loss.item() if isinstance(sparsity_loss, torch.Tensor) else sparsity_loss
            avg_loss_sememe += sememe_loss.item() if isinstance(sememe_loss, torch.Tensor) else sememe_loss

            # accuracy
            total_correct += correct
            total_correct_gap += correct_gap
            total_correct_discrete += correct_discrete
            total_correct_continuous += correct_continuous

            total_correct_mask += correct_mask
            total_correct_mask_gap += correct_mask_gap
            total_correct_mask_continuous += correct_mask_continuous
            total_correct_mask_discrete += correct_mask_discrete
            total_element += data["is_next"].nelement()
            total_element_mask += (data["bert_label"] != 0).sum().item()

            post_fix = {
                "epoch": epoch,
                "iter": i,
                "avg_loss": avg_loss / (i + 1),
                "avg_loss_next": avg_loss_next / (i + 1),
                "avg_loss_mask": avg_loss_mask / (i + 1),
                "avg_loss_rc": avg_loss_rc / (i + 1),
                "avg_loss_next_discrete": avg_loss_next_discrete / (i + 1),
                "avg_loss_mask_discrete": avg_loss_mask_discrete / (i + 1),
                "avg_loss_next_continuous": avg_loss_next_continuous / (i + 1),
                "avg_loss_mask_continuous": avg_loss_mask_continuous / (i + 1),
                "avg_loss_next_gap": avg_loss_next_gap / (i + 1),
                "avg_loss_mask_gap": avg_loss_mask_gap / (i + 1),
                "avg_loss_entropy ": avg_loss_entropy  / (i + 1),
                "avg_loss_sparse": avg_loss_sparse / (i + 1),
                "avg_loss_sememe": avg_loss_sememe / (i + 1),

                "avg_acc_next": total_correct / total_element * 100,
                "avg_acc_next_gap": total_correct_gap / total_element * 100,
                "avg_acc_next_discrete": total_correct_discrete / total_element * 100,
                "avg_acc_next_continuous": total_correct_continuous / total_element * 100,
                "avg_acc_mask": total_correct_mask / total_element_mask * 100,
                "avg_acc_mask_gap": total_correct_mask_gap / total_element_mask * 100,
                "avg_acc_mask_discrete": total_correct_mask_discrete / total_element_mask * 100,
                "avg_acc_mask_continuous": total_correct_mask_continuous / total_element_mask * 100,

                "loss": loss.item(),
                "loss_next": next_loss.item(),
                "loss_mask": mask_loss.item(),
                "loss_rc": rc_loss.item() if isinstance(rc_loss, torch.Tensor) else rc_loss,
                "loss_next_discrete": next_loss_discrete.item() if isinstance(next_loss_discrete, torch.Tensor) else next_loss_discrete,
                "loss_mask_discrete": mask_loss_discrete.item() if isinstance(mask_loss_discrete, torch.Tensor) else mask_loss_discrete,
                "loss_next_continuous": next_loss_continuous.item() if isinstance(next_loss_continuous, torch.Tensor) else next_loss_continuous,
                "loss_mask_continuous": mask_loss_continuous.item() if isinstance(mask_loss_continuous, torch.Tensor) else mask_loss_continuous,
                "loss_next_gap": next_loss_gap.item() if isinstance(next_loss_gap, torch.Tensor) else next_loss_gap,
                "loss_mask_gap": mask_loss_gap.item() if isinstance(mask_loss_gap, torch.Tensor) else mask_loss_gap,
                "loss_entropy ": entropy_loss.item() if isinstance(entropy_loss, torch.Tensor) else entropy_loss,
                "loss_sparse": sparsity_loss.item() if isinstance(sparsity_loss, torch.Tensor) else sparsity_loss,
                "loss_sememe": sememe_loss.item() if isinstance(sememe_loss, torch.Tensor) else sememe_loss,

                "lr": self.optim.param_groups[0]['lr']
            }

            if i % self.log_freq == 0:
                data_iter.write(str(post_fix))

        print("EP%d_%s, avg_loss=" % (epoch, str_code), avg_loss / len(data_iter), "total_acc=",
              total_correct * 100.0 / total_element)

    def save(self, epoch, file_path="output/bert_trained.model"):
        """
        Saving the current BERT model on file_path

        :param epoch: current epoch number
        :param file_path: model output path which gonna be file_path+"ep%d" % epoch
        :return: final_output_path
        """
        output_path = file_path + ".ep%d" % epoch
        torch.save(self.bert.cpu().state_dict(), output_path)
        self.bert.to(self.device)
        print("EP:%d Model Saved on:" % epoch, output_path)
        return output_path
