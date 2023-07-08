import os
from os.path import dirname
from pathlib import Path

import numpy as np
import torch
from torch.utils import data

class MultiModalDataset(data.Dataset):

    def __init__(self, individuals_file, directory):
        self.directory = directory
        self.individuals = np.loadtxt(os.path.join(directory, individuals_file), delimiter=",", dtype="float32")
        
    def __len__(self):
        return np.size(self.individuals, axis=0)

    def __getitem__(self, idx):
        individual = self.individuals[idx]
        indication = individual[1]
        values = individual[1:]
        return values, indication
    
    def __getIndividuals__(self):
        return self.individuals