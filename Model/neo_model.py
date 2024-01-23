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

class NEO_NN(nn.Module):

    def __init__(self, input_size, output_size, batch):
        super().__init__()

        self.l1 = nn.Linear(input_size, 30) 
        self.l2 = nn.Linear(30, output_size)
        
        self.relu1 = nn.ReLU()
        self.sig   = nn.Sigmoid()


    def forward(self, x):
        x = self.relu1(self.l1(x))
        out = self.sig(self.l2(x))
        
        return out
