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

        self.fc = nn.Linear(dv, 1)

        self.dropout = nn.Dropout(attn_dropout)

        self.batch = batch

    def forward(self, q, k, v):

        q_res = self.W_q(q)
        k_res = self.W_k(k)
        v_res = self.W_v(v)

        k_T = k_res.transpose(1,2)

        res = torch.matmul(q_res / math.sqrt(self.dk), k_T)
        attn_weights = F.softmax(res, dim=-1)

        out = torch.matmul(attn_weights, v_res)

        result = self.fc(out)

        reshaped = torch.reshape(result, [self.batch,60])

        return reshaped, attn_weights

        