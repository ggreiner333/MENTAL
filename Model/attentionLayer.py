import os
from os.path import dirname
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

import torch
import torch.nn as nn

import mne
import math


##################################################################################################
##################################################################################################
##################################################################################################

class AttentionLayer(nn.Module):

    def __init__(self, dk, dv, batch):
        super().__init__()

        self.dk = dk
        self.dv = dv

        self.W_q = nn.Linear(dk, dk)
        self.W_k = nn.Linear(130, dk)
        self.W_v = nn.Linear(130, dv)

        self.softmax = nn.Softmax()

    def forward(self, q, k, v):
        #print("Q size: " + str(q.size()))
        #print("K size: " + str(k.size()))
        #print("V size: " + str(v.size()))

        q = self.W_q(q)
        k = self.W_k(k)
        v = self.W_v(v)

        #print("Q size: " + str(q.size()))
        #print("K size: " + str(k.size()))
        #print("V size: " + str(v.size()))

        k = k.transpose(0,1)

        res = torch.matmul(q,k) / math.sqrt(self.dk)
        res = self.softmax(res)

        out = torch.matmul(res, v)

        #print(out.size())

        return out, res

        