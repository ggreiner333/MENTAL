import os
from os.path import dirname
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

import torch
import torch.nn as nn

import mne


##################################################################################################
##################################################################################################
##################################################################################################

class RNN_EEG(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.layer_1 = nn.GRU(input_size, hidden_size, num_layers=2, batch_first=False)
        self.output = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.Sigmoid()
        )

    def forward(self, x, h):
        x.unsqueeze_(-1)

        x = x.transpose(1,2)
        #print(x.size())

        res, h_1 = self.layer_1(x, h)
        #res2, h_2 = self.layer_2(res, h[1])
        #print(res.shape)

        out = self.output(res)
        #real = out.squeeze(-1)
        #print(real)

        return out, h_1
        