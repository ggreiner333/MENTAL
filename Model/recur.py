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
        print(x.size())
        res, h1 = self.layer_1(  x, h[0])
        res, h2 = self.layer_2(res, h[1])
        return self.output(res), (h1, h2)
        