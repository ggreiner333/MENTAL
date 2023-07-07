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


#----------------------------------#
#       Size of input vector       #
#----------------------------------#
#       disorder | 1               #
#    demographic | 3               #
#        NEO-FFI | 60              #
#   PSD Features | 130 x 60 = 7800 #
# ---------------------------------#
#          total | 7864            #
#----------------------------------#

# Layers
#     Input: shape(7864, 1)
#     
 

class VAE(nn.Module):

    def __init__(self, input_dim, z_dim):

        super().__init__()

        # Dimensions of importance for architecture
        self.input_dim  = input_dim
        self.z_dim      = z_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.RELU(),
            nn.Linear(1024, 512),
            nn.RELU(),
            nn.Linear(512, 128)
        )

        self.decoder = nn.Sequential(
            nn.Linear(128, 512),
            nn.RELU(),
            nn.Linear(512, 1024),
            nn.RELU(),
            nn.Linear(1024, input_dim)
        )

    def forward(self, input):
        z = self.encoder(input)
        res = self.decoder(z) 
        return res
