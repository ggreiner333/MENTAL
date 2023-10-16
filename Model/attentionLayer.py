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

        self.W_q = nn.Linear(1, dk)
        self.W_k = nn.Linear(1, dk)
        self.W_v = nn.Linear(1, dv)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, q, k, v):
        print("Q size: " + str(q.size()))
        print("K size: " + str(k.size()))
        print("V size: " + str(v.size()))

        q_res = self.W_q(q)
        k_res = self.W_k(k)
        v_res = self.W_v(v)

        print("Q size: " + str(q_res.size()))
        print("K size: " + str(k_res.size()))
        print("V size: " + str(v_res.size()))

        k_T = k_res.transpose(0,1)

        res = torch.matmul(q_res,k_T) / math.sqrt(self.dk)
        attn_weights = self.softmax(res)

        out = torch.matmul(attn_weights, v_res)

        #print(out.size())

        return out, res

        