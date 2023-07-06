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
#    demographic | 3               #
#        NEO-FFI | 60              #
#   PSD Features | 130 x 60 = 7800 #
# ---------------------------------#
#          total | 7863            #
#----------------------------------#

# Layers
#     Input: shape(7863, 1)
#     
 

class VAE(nn.Module):

    def __init__(self, input_dim, hidden_dim, z_dim):

        super(VAE, self).__init_()

        # Dimensions of importance for architecture
        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim
        self.z_dim      = z_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.RELU(),
            nn.Linear(1024, 512),
            nn.RELU(),
            nn.Linear(512, 128),
            nn.RELU(),
            nn.Linear(128, 64)
        )

        self.decoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.RELU(),
            nn.Linear(128, 512),
            nn.RELU(),
            nn.Linear(512, 1024),
            nn.RELU(),
            nn.Linear(1024, input_dim)
        )

    def forward(self, input):
        res = self.encoder(input)
        
        return res
    
    def decoder(self, z):
        return self.decoder_net(z)
