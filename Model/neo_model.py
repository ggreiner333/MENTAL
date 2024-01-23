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

        self.l1 = nn.Linear(input_size, 45) 
        self.l2 = nn.Linear(45, 30)
        self.l3 = nn.Linear(30, 15)
        self.l4 = nn.Linear(15, output_size)
        
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()
        self.sig   = nn.Sigmoid()


    def forward(self, x):
        x = self.relu1(self.l1(x))
        x = self.relu2(self.l2(x))
        x = self.relu3(self.l3(x))
        out = self.sig(self.l4(x))
        
        return out
