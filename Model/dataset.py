import os
from os.path import dirname
from pathlib import Path

import numpy as np
import torch
from torch.utils import data

class MultiModalDataset(data.Dataset):
    
    def __init__(self, individuals_file, directory):
        self.directory = directory
        self.individuals = np.loadtxt(os.path.join(directory, individuals_file), delimiter=",", dtype=str)
        self.individuals = self.individuals[1:]
        
    def __len__(self):
        return np.size(self.individuals)-1

    def __getitem__(self, idx):
        individual = self.individuals[idx]
        indication = individual[2]
        values = individual[3:]
        return values, indication