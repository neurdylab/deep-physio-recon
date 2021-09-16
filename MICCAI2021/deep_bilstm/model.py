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

# # ''' attention model '''
# class BidirectionalLSTM(nn.Module):
#
#     def __init__(self, nCh, nHidden, nOut):
#         super(BidirectionalLSTM, self).__init__()
#
#         # attention block
#         self.rnn = nn.LSTM(nCh, nCh, bidirectional=False)
#
#         self.spatial = nn.Softmax(dim=1)
#         self.temporal = nn.Softmax(dim=0)
#
#         # self.spatial = nn.Sigmoid()
#         # self.temporal = nn.Sigmoid()
#
#         # self.rnn2 = nn.LSTM(nCh, nHidden, bidirectional=True)
#         # self.embedding = nn.Linear(nHidden, nOut) % LSTM
#         self.embedding = nn.Linear(nHidden, nOut)
#         self.embedding2 = nn.Linear(nHidden, nOut)
#
#     def forward(self, input):
#         # input of shape (seq_len, batch, input_size)
#         input = input.permute(2, 0, 1)
#         print("block input size = %s" % (str(input.size())))
#
#         low, _ = self.rnn(input)
#         T, b, h = low.size()
#         # print("block input size = %s, %s, %s" % (T, b, h))
#
#         s_mask_low = torch.sum(low, dim=0)
#         s_mask = self.spatial(s_mask_low)
#         # print("block s_mask size = %s" % (str(s_mask_low.size())))
#
#         t_mask_low = torch.sum(low, dim=2)
#         t_mask = self.temporal(t_mask_low)
#         # # print("block t_mask size = %s" % (str(t_mask_low.size())))
#         #
#         # att_input = input * s_mask
#         # # print("block att input size = %s" % (str(att_input.size())))
#         # recurrent, _ = self.rnn2(att_input)
#         # T, b, h = recurrent.size()
#         # # print("block input size = %s, %s, %s" % (T, b, h))
#         #
#         # att_recurrent = recurrent * t_mask.unsqueeze(-1).repeat(1,1,h)
#
#         # print("block input size = %s, %s, %s" % (T, b, h))
#         t_rec = low.reshape(T * b, h)
#         # print(input.shape, recurrent.shape, t_rec.shape)
#
#         out1 = self.embedding(t_rec)  # [T * b, nOut]
#         # print("block output size = %s" % (str(out1.size())))
#         out1 = out1.view(T, b, -1)
#         # print("block output size = %s" % (str(out.size())))
#         out1 = out1.permute(1, 2, 0)
#
#         out2 = self.embedding2(t_rec)  # [T * b, nOut]
#         out2 = out2.view(T, b, -1)
#         # print("block output size = %s" % (str(out2.size())))
#         out2 = out2.permute(1, 2, 0)
#         return out1, out2, t_mask, s_mask

''' IPMI model'''
class BidirectionalLSTM(nn.Module):

    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM, self).__init__()

        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        # self.embedding = nn.Linear(nHidden, nOut) % LSTM
        self.embedding = nn.Linear(nHidden*2, nOut)
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

        out = self.embedding(t_rec)  # [T * b, nOut]
        out = out.view(T, b, -1)
        # print("block output size = %s" % (str(out.size())))
        out = out.permute(1, 2, 0)

        out2 = self.embedding2(t_rec)  # [T * b, nOut]
        out2 = out2.view(T, b, -1)
        # print("block output size = %s" % (str(out2.size())))
        out2 = out2.permute(1, 2, 0)
        return out, out2
