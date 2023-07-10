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

    def __init__(self, dk, dv):
        super().__init__()

        self.dk = dk
        self.dv = dv

        self.W_q = nn.Linear(1, dk)
        self.W_k = nn.Linear(1, dk)
        self.W_v = nn.Linear(1, dv)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v):
        q = self.W_q(q)
        k = self.W_k(k)
        v = self.W_v(v)

        k = k.transpose(2,3)

        res = torch.matmul(q,k) / math.sqrt(self.dk)
        res = self.softmax(res)

        out = torch.matmul(res, v)

        return out, res

        