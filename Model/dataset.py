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
        output = torch.zeros([1])
        if(int(indication) == 2):
            output[0] = 1
        else:
            output[0] = 0

        dem_val = individual[2:5]
        #dem_out = np.zeros(30, dtype="float32")
        #dem_out[0] = dem_val[0]
        #dem_out[1] = dem_val[1]
        #dem_out[2] = dem_val[2]
        #dem_out = torch.from_numpy(dem_out)

        h_1 = np.zeros(30, dtype="float32")
        h_2 = np.zeros(15, dtype="float32")

        h_1t = torch.from_numpy(h_1)
        h_2t = torch.from_numpy(h_2)

        h_out = (h_1t, h_2t)

        neo_val = individual[5:65]
        print(neo_val)
        print("4: " + str(individual[4]))

        print("64: " + str(individual[64]))
        print("65: " + str(individual[65]))
        psd_val = []
        for i in range(0,7800, 130):
            psd_val.append(individual[i:(i+130)]) 

        return h_out, neo_val, psd_val, output
    
    def __getIndividuals__(self):
        res = []
        for i in range(0, self.__len__()):
            res.append((self.__getitem__(i))[0])
        return np.array(res)
    

class MSplitDataset(data.Dataset):

    def __init__(self, individuals_file, directory):
        self.directory = directory
        self.individuals = np.loadtxt(os.path.join(directory, individuals_file), delimiter=",", dtype="float32")
        
    def __len__(self):
        return np.size(self.individuals, axis=0)

    def __getitem__(self, idx):
        individual = self.individuals[idx]

        indication = individual[1]
        output = torch.zeros([5])
        if(int(indication) == 2):
            output[1] = 1
        else:
            output[0] = 1

        dem_val = individual[2:5]
        dem_out = np.zeros(5, dtype="float32")
        dem_out[0] = dem_val[0]
        dem_out[1] = dem_val[1]
        dem_out[2] = dem_val[2]
        dem_out = torch.from_numpy(dem_out)

        neo_val = individual[5:65]

        psd_val = []
        for i in range(0,300, 5):
            psd_val.append(individual[i:(i+5)]) 

        return dem_out, neo_val, psd_val, output
    
    def __getIndividuals__(self):
        res = []
        for i in range(0, self.__len__()):
            res.append((self.__getitem__(i))[0])
        return np.array(res)
