import os
from os.path import dirname
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

import torch
import torch.nn as nn

import mne

from Model.attentionLayer import AttentionLayer
from Model.recur import EegRNN


##################################################################################################
##################################################################################################
##################################################################################################

class MENTAL(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):

        self.attention = AttentionLayer(60, 60)

        self.rnn = EegRNN(input_size, hidden_size, output_size)


    def forward(self, eeg, neo, h):

        # Run attention layer
        out, res = self.attention(neo, eeg, eeg)

        return self.rnn(out, h)
