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

class EegRNN(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.layer_1 = nn.GRU(input_size, hidden_size, batch_first=True)
        self.layer_2 = nn.GRU(hidden_size, output_size, batch_first=True)
        self.output = nn.Sequential(
            nn.ReLU(),
            nn.Linear(output_size, 1)
        )

    def forward(self, x, h):
        x.unsqueeze_(-1)
        x = x.transpose(1,2)
        print(x.size())
        h[0].unsqueeze_(-1)
        h0 = h[0].transpose(1,2)
        h0 = h0.transpose(0,1)
        print(h0.size())
        h[1].unsqueeze_(-1)
        h1 = h[1].transpose(1,2)
        h1 = h1.transpose(0,1)
        h1 = h1.squeeze(-1)
        print(h1.size())
        res, h_1 = self.layer_1(  x, h0)
        res, h_2 = self.layer_2(res, h1)
        return self.output(res), (h_1, h_2)
        