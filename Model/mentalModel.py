import os
from os.path import dirname
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

import torch
import torch.nn as nn

import mne

from attentionLayer import AttentionLayer
from recur import EegRNN


##################################################################################################
##################################################################################################
##################################################################################################

class MENTAL(nn.Module):

    def __init__(self):

        self.attention = AttentionLayer()

        self.rnn = EegRNN()


    def forward():

        # Run attention layer

        # Run it through RNN

        pass
