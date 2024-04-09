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

        self.layer_1 = nn.GRU(input_size, hidden_size, num_layers=2, batch_first=True)

        self.o1 = nn.Linear(hidden_size, 20) 
        self.o2 = nn.Linear(20, 10)
        self.o3 = nn.Linear(10, output_size)
        
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.sig   = nn.Sigmoid()
        self.soft  = nn.Softmax(dim=0)

    def forward(self, x, h):
        x.unsqueeze_(-1)

        x = x.transpose(1,2)

        res, h_1 = self.layer_1(x, h)

        res = self.relu1(self.o1(res))
        res = self.relu2(self.o2(res))
        out = self.o3(res)

        out.squeeze_(1)

        return out, h_1
        