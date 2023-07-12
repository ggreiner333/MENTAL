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
        res = [(self.individuals[0])[1:]]
        for i in range(1, self.__len__()):
            res.append((self.individuals[i])[1:])
        return np.array(res)
    

class SplitDataset(data.Dataset):

    def __init__(self, individuals_file, directory):
        self.directory = directory
        self.individuals = np.loadtxt(os.path.join(directory, individuals_file), delimiter=",", dtype="float32")
        
    def __len__(self):
        return np.size(self.individuals, axis=0)

    def __getitem__(self, idx):
        individual = self.individuals[idx]
        indication = individual[1]
        output = torch.zeros([35])
        output[int(indication)] = 1
        dem_val = individual[2:5]
        neo_val = individual[5:65]
        psd_val = []
        for i in range(0,7800, 130):
            psd_val.append(individual[i:(i+130)]) 
        return dem_val, neo_val, psd_val, output
    
    def __getIndividuals__(self):
        res = []
        for i in range(0, self.__len__()):
            res.append((self.__getitem__(i))[0])
        return np.array(res)
