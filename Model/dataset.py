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
        self.individuals = np.load(os.path.join(directory, individuals_file))
        
    def __len__(self):
        return np.size(self.individuals, axis=0)

    def __getitem__(self, idx):
        individual = self.individuals[idx]

        indication = individual[1]
        output = torch.zeros([1], dtype=torch.float32)
        if(int(indication) == 3):
            output[0] = 1
        else:
            output[0] = 0

        dem_val = individual[2:5]
        #dem_out = np.zeros(30, dtype="float32")
        #dem_out[0] = dem_val[0]
        #dem_out[1] = dem_val[1]
        #dem_out[2] = dem_val[2]
        #dem_out = torch.from_numpy(dem_out)

        h_1 = torch.zeros([2, 1, 30], dtype=torch.float32)
        #h_2 = np.zeros(15, dtype="float32")

        #h_1t = torch.from_numpy(h_1)
        #h_2t = torch.from_numpy(h_2)

        #h_out = (h_1t, h_2t)

        neo = individual[5:65]
        neo_val = torch.tensor(neo, dtype=torch.float32)
        neo = torch.reshape(neo_val, [60, 1])
        #print(neo.shape)

        psd = []
        for i in range(65,7865, 130):
            psd.append(individual[i:(i+130)])
        
        psd_val = torch.tensor(psd, dtype=torch.float32)
        psd = torch.reshape(psd_val, [60,130,1])
        #print(psd.shape)

        return h_1, neo, psd, output
    
    def __getIndividuals__(self):
        res = []
        for i in range(0, self.__len__()):
            res.append((self.__getitem__(i))[0])
        return np.array(res)
    
class BSplitDataset(data.Dataset):

    def __init__(self, individuals_file, directory):
        self.directory = directory
        self.individuals = np.load(os.path.join(directory, individuals_file))
        
    def __len__(self):
        return np.size(self.individuals, axis=0)

    def __getitem__(self, idx):
        individual = self.individuals[idx]

        indication = individual[1]
        output = torch.zeros([1], dtype=torch.float32)
        if(int(indication) == 3):
            output[0] = 1
        else:
            output[0] = 0

        dem_val = individual[2:5]
        #dem_out = np.zeros(30, dtype="float32")
        #dem_out[0] = dem_val[0]
        #dem_out[1] = dem_val[1]
        #dem_out[2] = dem_val[2]
        #dem_out = torch.from_numpy(dem_out)

        h_1 = torch.zeros([2, 1, 30], dtype=torch.float32)
        #h_2 = np.zeros(15, dtype="float32")

        #h_1t = torch.from_numpy(h_1)
        #h_2t = torch.from_numpy(h_2)

        #h_out = (h_1t, h_2t)

        neo = individual[5:65]
        neo_val = torch.tensor(neo, dtype=torch.float32)
        neo = torch.reshape(neo_val, [60, 1])
        #print(neo.shape)

        psd = []
        for i in range(65,15665, 130):
            psd.append(individual[i:(i+130)])
        
        psd_val = torch.tensor(psd, dtype=torch.float32)
        psd = torch.reshape(psd_val, [120,130, 1])
        #print(psd.shape)

        return h_1, neo, psd, output
    
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

class EEGDataset(data.Dataset):

    def __init__(self, individuals_file, directory):
        self.directory = directory
        self.individuals = np.load(os.path.join(directory, individuals_file))
        
    def __len__(self):
        return np.size(self.individuals, axis=0)

    def __getitem__(self, idx):
        individual = self.individuals[idx]

        indication = individual[0]
        output = torch.zeros([1], dtype=torch.float32)
        if(int(indication) == 3):
            output[0] = 1
        else:
            output[0] = 0

        h_1 = torch.zeros([2, 1, 30], dtype=torch.float32)

        psd = individual[1:]
        
        psd_val = torch.tensor(psd, dtype=torch.float32)
        psd = torch.reshape(psd_val, [60,130])

        return h_1, psd, output
    
    def __getIndividuals__(self):
        res = []
        for i in range(0, self.__len__()):
            res.append((self.__getitem__(i))[0])
        return np.array(res)

class EEGBothDataset(data.Dataset):

    def __init__(self, individuals_file, directory):
        self.directory = directory
        self.individuals = np.load(os.path.join(directory, individuals_file))
        
    def __len__(self):
        return np.size(self.individuals, axis=0)

    def __getitem__(self, idx):
        individual = self.individuals[idx]

        indication = individual[0]
        output = torch.zeros([1], dtype=torch.float32)
        if(int(indication) == 3):
            output[0] = 1
        else:
            output[0] = 0

        h_1 = torch.zeros([2, 1, 30], dtype=torch.float32)

        psd = individual[1:]
        
        psd_val = torch.tensor(psd, dtype=torch.float32)
        psd = torch.reshape(psd_val, [120,130])

        return h_1, psd, output
    
    def __getIndividuals__(self):
        res = []
        for i in range(0, self.__len__()):
            res.append((self.__getitem__(i))[0])
        return np.array(res)
    
class NEODataset(data.Dataset):

    def __init__(self, individuals_file, directory):
        self.directory = directory
        self.individuals = np.load(os.path.join(directory, individuals_file))
        
    def __len__(self):
        return np.size(self.individuals, axis=0)

    def __getitem__(self, idx):
        individual = self.individuals[idx]

        indication = individual[0]
        output = torch.zeros([1], dtype=torch.float32)
        if(int(indication) == 2):
            output[0] = 1
        else:
            output[0] = 0

        neo = individual[1:]
        
        neo_val = torch.tensor(neo, dtype=torch.float32)

        return neo_val, output
    
    def __getIndividuals__(self):
        res = []
        for i in range(0, self.__len__()):
            res.append((self.__getitem__(i))[0])
        return np.array(res)