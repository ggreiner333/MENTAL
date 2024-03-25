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

        self.encode1 = nn.Linear(input_dim, 3000)
        self.encode2 = nn.Linear(3000, 1000)

        self.mu  = nn.Linear(1024, z_dim)
        self.var = nn.Linear(1024, z_dim)

        self.decode1 = nn.Linear(z_dim, 1024)
        self.decode2 = nn.Linear(1024, 3000)
        self.decode3 = nn.Linear(3000, input_dim)

    def encode(self, x):
        res = F.relu(self.encode1(x  ))
        res = F.relu(self.encode2(res))

        Mu  = F.sigmoid(self.mu(res))
        Var = F.sigmoid(self.var(res))
        
        return Mu, Var
    
    def decode(self, z):
        res = F.relu(self.decode1(z  ))
        res = F.relu(self.decode2(res))
        res = F.relu(self.decode3(res))

        print("hi")

        return res

    def forward(self, x):
        Mu, Var = self.encode(x)

        dist = torch.distributions.normal.Normal(Mu, torch.exp(0.5*Var))
        sample = dist.rsample()

        out = self.decode(sample)

        return out, Mu, Var
