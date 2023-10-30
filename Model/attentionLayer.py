import os
from os.path import dirname
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

import mne
import math


##################################################################################################
##################################################################################################
##################################################################################################

class AttentionLayer(nn.Module):

    def __init__(self, dk, dv, batch, attn_dropout=0.1):
        super().__init__()

        self.dk = dk
        self.dv = dv

        self.W_q = nn.Linear(1, dk, bias = False)
        self.W_k = nn.Linear(1, dk, bias = False)
        self.W_v = nn.Linear(1, dv, bias = False)

        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v):
        #print("Q size: " + str(q.size()))
        #print("K size: " + str(k.size()))
        #print("V size: " + str(v.size()))

        q_res = self.W_q(q)
        k_res = self.W_k(k)
        v_res = self.W_v(v)

        #print("Q size: " + str(q_res.size()))
        #print("K size: " + str(k_res.size()))
        #print("V size: " + str(v_res.size()))

        #k_T = k_res.transpose(0,1)

        print("Q size: " + str(q_res.size()))
        print("K size: " + str(k.size()))
        print("V size: " + str(v_res.size()))

        res = torch.matmul(q_res / math.sqrt(self.dk), k)
        attn_weights = self.dropout(F.softmax(res, dim=-1))

        out = torch.matmul(attn_weights, v_res)

        #print(out.size())

        return out, attn_weights

        