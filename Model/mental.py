import os
from os.path import dirname
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

import torch
import torch.nn as nn

import mne

from Model.recurEEG import RNN_EEG

##################################################################################################
##################################################################################################
##################################################################################################

class MENTAL_EEG(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, batch):
        super().__init__()

        self.rnn = RNN_EEG(input_size, hidden_size, output_size)


    def forward(self, eeg, h):

        rnn_out = self.rnn(eeg, h)

        #print(rnn_out)

        return rnn_out