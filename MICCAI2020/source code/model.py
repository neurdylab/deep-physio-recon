

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import random

seed_num = 5
torch.manual_seed(seed_num)
random.seed(seed_num)

class linear(nn.Module):
    def __init__(self, in_chs, out_chs, opt):
        super(linear, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv1d(in_chs, 1, kernel_size=1, stride=1, padding=0))

    def forward(self, x):
        # print("block in size = %s" % (str(x.size())))
        x = self.conv1(x)
        # print("block out size = %s" % (str(x.size())))
        return x



class UNet(nn.Module):
    def __init__(self, in_channel, out_channel, opt):
        super(UNet, self).__init__()

        self.ec0 = self.encoder(in_channel, 32, bias=True, batchnorm=True)
        self.ec1 = self.encoder(32, 64, bias=True, batchnorm=True)
        self.ec2 = self.encoder(64, 64, bias=True, batchnorm=True)
        self.ec3 = self.encoder(64, 128, bias=True, batchnorm=True)
        self.ec4 = self.encoder(128, 128, bias=True, batchnorm=True)
        self.ec5 = self.encoder(128, 256, bias=True, batchnorm=True)
        self.ec6 = self.encoder(256, 256, bias=True, batchnorm=True)
        self.ec7 = self.encoder(256, 512, bias=True, batchnorm=True)

        # random comment to test git

        self.pool0 = nn.MaxPool1d(2)
        self.pool1 = nn.MaxPool1d(2)
        self.pool2 = nn.MaxPool1d(2)

        self.dc9 = self.decoder(512, 512, kernel_size=2, stride=2, bias=True)
        self.dc8 = self.decoder(256 + 512, 256, kernel_size=3, stride=1, padding=1, bias=True)
        self.dc7 = self.decoder(256, 256, kernel_size=3, stride=1, padding=1, bias=True)
        self.dc6 = self.decoder(256, 256, kernel_size=2, stride=2, bias=True)
        self.dc5 = self.decoder(128 + 256, 133, kernel_size=3, stride=1, padding=1, bias=True)
        self.dc4 = self.decoder(133, 133, kernel_size=3, stride=1, padding=1, bias=True)
        self.dc3 = self.decoder(133, 133, kernel_size=2, stride=2, bias=True)
        self.dc2 = self.decoder(64 + 133, 133, kernel_size=3, stride=1, padding=1, bias=True)
        self.dc1 = self.decoder(133, 133, kernel_size=3, stride=1, padding=1, bias=True)
        self.dc0 = self.decoder(133, 1, kernel_size=1, stride=1, bias=True)

    def encoder(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                bias=True, batchnorm=False):
        if batchnorm:
            layer = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias))
        else:
            layer = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias))
        return layer


    def decoder(self, in_channels, out_channels, kernel_size, stride=1, padding=0 ,
                output_padding=0, bias=True):
        layer = nn.Sequential(
            nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride=stride,
                               padding=padding, output_padding=output_padding, bias=bias))
        return layer

    def forward(self, x):
        e0 = self.ec0(x)
        syn0 = self.ec1(e0)
        e1 = self.pool0(syn0)
        e2 = self.ec2(e1)
        syn1 = self.ec3(e2)
        del e0, e1, e2

        e3 = self.pool1(syn1)
        e4 = self.ec4(e3)
        syn2 = self.ec5(e4)
        del e3, e4

        e5 = self.pool2(syn2)
        e6 = self.ec6(e5)
        e7 = self.ec7(e6)
        del e5, e6

        # print("block e7 size = %s" % (str(e7.size())))
        # print("block dc9 size = %s" % (str(self.dc9(e7).size())))
        # print("block syn2 size = %s" % (str(syn2.size())))
        d9 = torch.cat((self.dc9(e7), syn2), 1)
        # print("block d9 size = %s" % (str(d9.size())))
        del e7, syn2

        d8 = self.dc8(d9)
        d7 = self.dc7(d8)
        # print("block d8 size = %s" % (str(d8.size())))
        del d9, d8
        # print("block d7 size = %s" % (str(d7.size())))
        d6 = torch.cat((self.dc6(d7), syn1), 1)
        del d7, syn1

        d5 = self.dc5(d6)
        d4 = self.dc4(d5)
        # print("block d5 size = %s" % (str(d5.size())))
        # print("block d4 size = %s" % (str(d4.size())))
        del d6, d5

        d3 = torch.cat((self.dc3(d4), syn0), 1)
        del d4, syn0
        # print("block d3 size = %s" % (str(d3.size())))

        d2 = self.dc2(d3)
        d1 = self.dc1(d2)
        # print("block d2 size = %s" % (str(d2.size())))
        del d3, d2
        # print("block d1 size = %s" % (str(d1.size())))

        d0 = self.dc0(d1)
        # print("block d0 size = %s" % (str(d0.size())))

        return d0


class BidirectionalLSTM(nn.Module):

    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM, self).__init__()

        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        # self.embedding = nn.Linear(nHidden, nOut)
        self.embedding = nn.Linear(nHidden*2, nOut)


    def forward(self, input):
        # input of shape (seq_len, batch, input_size)
        input = input.permute(2, 0, 1)
        # print("block input size = %s" % (str(input.size())))

        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size()
        # print("block input size = %s, %s, %s" % (T, b, h))
        t_rec = recurrent.view(T * b, h)
        # print(input.shape, recurrent.shape, t_rec.shape)

        output = self.embedding(t_rec)  # [T * b, nOut]
        output = output.view(T, b, -1)
        # print("block output size = %s" % (str(output.size())))
        output = output.permute(1, 2, 0)
        return output
