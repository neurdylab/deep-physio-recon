import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import random
import copy
from typing import Optional, Any

from torch import Tensor
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_

seed_num = 5
torch.manual_seed(seed_num)
random.seed(seed_num)

class BidirectionalLSTM(nn.Module):

    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM, self).__init__()

        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        # self.embedding = nn.Linear(nHidden, nOut) % LSTM
        self.embedding = nn.Linear(nHidden*2, nOut)
        self.drop = nn.Dropout(p=0.3)
        self.embedding2 = nn.Linear(nHidden*2, nOut)

    def forward(self, input):
        # input of shape (seq_len, batch, input_size)
        input = input.permute(2, 0, 1)
        # print("block input size = %s" % (str(input.size())))

        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size()
        # print("block input size = %s, %s, %s" % (T, b, h))
        t_rec = recurrent.view(T * b, h)
        # print(input.shape, recurrent.shape, t_rec.shape)

        t_rec = self.drop(t_rec)  # dropout

        out = self.embedding(t_rec)  # [T * b, nOut]
        out = out.view(T, b, -1)
        # print("block output size = %s" % (str(out.size())))
        out = out.permute(1, 2, 0)

        out2 = self.embedding2(t_rec)  # [T * b, nOut]
        out2 = out2.view(T, b, -1)
        # print("block output size = %s" % (str(out2.size())))
        out2 = out2.permute(1, 2, 0)
        return out, out2