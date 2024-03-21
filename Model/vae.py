import os
from os.path import dirname
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

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

        self.mu  = nn.Linear(512, z_dim)
        self.var = nn.Linear(512, z_dim)

        self.decode1 = nn.Linear(z_dim, 512)
        self.decode2 = nn.Linear(512, 1024)
        self.decode3 = nn.Linear(1024, input_dim)

    def encode(self, x):
        res = F.relu(self.encode1(x  ))
        res = F.relu(self.encode2(res))

        mu  = F.relu(self.mu(res))
        var = F.relu(self.var(res))

        return mu, var
    
    def decode(self, z):
        res = F.relu(self.decode1(z  ))
        res = F.relu(self.decode2(res))
        res = self.decode3(res)

        return res

    def forward(self, x):
        mu, var = self.encode(x)

        dist = torch.distributions.normal.Normal(mu, torch.exp(0.5*var))
        sample = dist.rsample()

        out = self.decode(sample)

        return out, mu, var
