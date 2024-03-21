import os
from os.path import dirname
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

import torch
import torch.nn as nn


##################################################################################################
##################################################################################################
##################################################################################################

 

class VAE(nn.Module):

    def __init__(self, input_dim, z_dim):

        super().__init__()

        # Dimensions of importance for architecture
        self.input_dim  = input_dim
        self.z_dim      = z_dim

        self.encode1 = nn.Linear(input_dim, 1024)
        self.encode2 = nn.Linear(1024, 512)
        self.encode3 = nn.Linear(512, 256)

        self.mu  = nn.Linear(256, z_dim)
        self.var = nn.Linear(256, z_dim)

        self.decode1 = nn.Linear(z_dim, 256)
        self.decode2 = nn.Linear(256, 512)
        self.decode3 = nn.Linear(512, 1024)
        self.decode4 = nn.Linear(1024, input_dim)

    def encode(self, x):
        res = nn.ReLU(self.encode1(x  ))
        res = nn.ReLU(self.encode2(res))
        res = nn.ReLU(self.encode3(res))

        mu  = self.mu(res)
        var = self.var(res)

        return mu, var
    
    def decode(self, z):
        res = nn.ReLU(self.decode1(z  ))
        res = nn.ReLU(self.decode2(res))
        res = nn.ReLU(self.decode3(res))
        res = nn.ReLU(self.decode4(res))

        return res

    def forward(self, x):
        mu, var = self.encode(x)

        dist = torch.distributions.normal.Normal(mu, torch.exp(0.5*var))
        sample = dist.rsample()

        out = self.decode(sample)

        return out, mu, var
