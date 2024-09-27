import os
from os.path import dirname
from pathlib import Path

import numpy as np
import torch
from torch.utils import data

import mne
import math

import matplotlib.pyplot as plt

from Model.dataset import MultiModalDataset
from Model.dataset import SplitDataset
from Model.dataset import BSplitDataset
from Model.dataset import MSplitDataset
from Model.dataset import M3SplitDataset
from Model.mentalModel import MENTAL
from Model.mental import MENTAL_EEG


##################################################################################################
##################################################################################################
##################################################################################################

def run_train_EC(learn_rate, wd, batch_sz, epochs, outfile):

    main_dataset = SplitDataset('normalized_small_complete_samples_EC_health_mdd.npy', '/data/zhanglab/ggreiner/MENTAL/TDBRAIN')

    print(main_dataset.__len__())

    splits = [60, 35, 1]

    res = data.random_split(main_dataset, splits)

    train_loader = data.DataLoader(res[0], batch_size=batch_sz, shuffle=True)
    test_loader  = data.DataLoader(res[1], batch_size=batch_sz, shuffle=True)

    my_mental = MENTAL(60, 30, 1, batch_sz)

    optimizer = torch.optim.Adam(my_mental.parameters(), lr=learn_rate)

    torch.autograd.set_detect_anomaly(True)

    preds = []
    conds = []

    for epoch in range(epochs):
        print(epoch)
        for (h_entry, n_entry, p_entry, label) in train_loader:

            test = []
            for i in range(0, 60):
                batch = []
                for j in range(0,batch_sz):
                    cur = p_entry[j][i]
                    arr_cur = np.asarray(cur)
                    batch.append(arr_cur)
                test.append(batch)

            formatted = np.array(test)
            psd_tensor = torch.from_numpy(formatted)
            
            h_1 = torch.zeros([2, batch_sz, 30], dtype=torch.float32)

            for p in psd_tensor:
                output, h_res = my_mental.forward(p, n_entry, h_1)
                h = h_res

            sig = torch.nn.Sigmoid()
            output = sig(output)

            loss = torch.nn.BCELoss()
            res = loss(output, label)

            optimizer.zero_grad()
            res.backward()
            optimizer.step()
        
        if(epoch == (epochs-1)):
            for (h_entry, n_entry, p_entry, label) in test_loader:

                test = []
                for i in range(0, 60):
                    batch = []
                    for j in range(0,batch_sz):
                        cur = p_entry[j][i]
                        arr_cur = np.asarray(cur)
                        batch.append(arr_cur)
                    test.append(batch)

                formatted = np.array(test)
                psd_tensor = torch.from_numpy(formatted)
                
                h_1 = torch.zeros([2, batch_sz, 30], dtype=torch.float32)
                
                for p in psd_tensor:
                    output, h_res = my_mental.forward(p, n_entry, h_1)
                    h = h_res

                sig = torch.nn.Sigmoid()
                output = sig(output)

                out = output.squeeze_(1)
                for i in range(0, batch_sz):
                    preds.append(out[i].detach())

                label = label.squeeze_(1)
                for i in range(0, len(label)):
                    conds.append(label[i].item())

 
    preds = np.array(preds)
    print(preds)
    conds = np.array(conds)
    print(conds)

    np.save('/home/ggreiner/MENTAL/roc2/MENTAL_EC_HEALTH_MDD_PREDICTIONS', preds)
    np.save('/home/ggreiner/MENTAL/roc2/MENTAL_EC_HEALTH_MDD_CONDITIONS', conds)

def run_train_EO(learn_rate, wd, batch_sz, epochs, outfile):

    main_dataset = SplitDataset('normalized_small_complete_samples_EO_health_mdd.npy', '/data/zhanglab/ggreiner/MENTAL/TDBRAIN')

    splits = [60, 35,1]

    res = data.random_split(main_dataset, splits)

    train_loader = data.DataLoader(res[0], batch_size=batch_sz, shuffle=True)
    test_loader  = data.DataLoader(res[1], batch_size=batch_sz, shuffle=True)

    my_mental = MENTAL(60, 30, 1, batch_sz)


    optimizer = torch.optim.Adam(my_mental.parameters(), lr=learn_rate)

    torch.autograd.set_detect_anomaly(True)

    preds = []
    conds = []

    for epoch in range(epochs):
        print(epoch)
        for (h_entry, n_entry, p_entry, label) in train_loader:

            test = []
            for i in range(0, 60):
                batch = []
                for j in range(0,batch_sz):
                    cur = p_entry[j][i]
                    arr_cur = np.asarray(cur)
                    batch.append(arr_cur)
                test.append(batch)

            formatted = np.array(test)
            psd_tensor = torch.from_numpy(formatted)
            
            h_1 = torch.zeros([2, batch_sz, 30], dtype=torch.float32)

            for p in psd_tensor:
                output, h_res = my_mental.forward(p, n_entry, h_1)
                h = h_res

            sig = torch.nn.Sigmoid()
            output = sig(output)

            loss = torch.nn.BCELoss()
            res = loss(output, label)

            optimizer.zero_grad()
            res.backward()
            optimizer.step()
        
        if(epoch == (epochs-1)):
            for (h_entry, n_entry, p_entry, label) in test_loader:

                test = []
                for i in range(0, 60):
                    batch = []
                    for j in range(0,batch_sz):
                        cur = p_entry[j][i]
                        arr_cur = np.asarray(cur)
                        batch.append(arr_cur)
                    test.append(batch)

                formatted = np.array(test)
                psd_tensor = torch.from_numpy(formatted)
                
                h_1 = torch.zeros([2, batch_sz, 30], dtype=torch.float32)
                
                for p in psd_tensor:
                    output, h_res = my_mental.forward(p, n_entry, h_1)
                    h = h_res

                sig = torch.nn.Sigmoid()
                output = sig(output)

                out = output.squeeze_(1)
                for i in range(0, batch_sz):
                    preds.append(out[i].detach())

                label = label.squeeze_(1)
                for i in range(0, len(label)):
                    conds.append(label[i].item())

 
    preds = np.array(preds)
    print(preds)
    conds = np.array(conds)
    print(conds)

    np.save('/home/ggreiner/MENTAL/roc2/MENTAL_EO_HEALTH_MDD_PREDICTIONS', preds)
    np.save('/home/ggreiner/MENTAL/roc2/MENTAL_EO_HEALTH_MDD_CONDITIONS', conds)

def run_train_both(learn_rate, wd, batch_sz, epochs, outfile):

    main_dataset = BSplitDataset('normalized_small_imputed_complete_samples_EC_EO_depression.npy', '/data/zhanglab/ggreiner/MENTAL/TDBRAIN')

    splits = []
    splits = [735,225,8]
    """
    if(batch_sz != 15):
        splits = [560, 140, 4]
    else:
        splits = [540, 150, 14]
    """

    res = data.random_split(main_dataset, splits)

    train_loader = data.DataLoader(res[0], batch_size=batch_sz, shuffle=True)
    test_loader  = data.DataLoader(res[1], batch_size=batch_sz, shuffle=True)

    my_mental = MENTAL(60, 30, 1, batch_sz)

    optimizer = torch.optim.Adam(my_mental.parameters(), lr=learn_rate, weight_decay=wd)

    torch.autograd.set_detect_anomaly(True)

    preds = []
    conds = []

    for epoch in range(epochs):
        print(epoch)
        for (h_entry, n_entry, p_entry, label) in train_loader:

            test = []
            for i in range(0, 120):
                batch = []
                for j in range(0,batch_sz):
                    cur = p_entry[j][i]
                    arr_cur = np.asarray(cur)
                    batch.append(arr_cur)
                test.append(batch)

            formatted = np.array(test)
            psd_tensor = torch.from_numpy(formatted)
            
            h_1 = torch.zeros([2, batch_sz, 30], dtype=torch.float32)

            for p in psd_tensor:
                output, h_res = my_mental.forward(p, n_entry, h_1)
                h = h_res

            sig = torch.nn.Sigmoid()
            output = sig(output)

            loss = torch.nn.BCELoss()
            res = loss(output, label)

            optimizer.zero_grad()
            res.backward()
            optimizer.step()
        
        if(epoch == (epochs-1)):
            for (h_entry, n_entry, p_entry, label) in test_loader:

                test = []
                for i in range(0, 120):
                    batch = []
                    for j in range(0,batch_sz):
                        cur = p_entry[j][i]
                        arr_cur = np.asarray(cur)
                        batch.append(arr_cur)
                    test.append(batch)

                formatted = np.array(test)
                psd_tensor = torch.from_numpy(formatted)
                
                h_1 = torch.zeros([2, batch_sz, 30], dtype=torch.float32)
                
                for p in psd_tensor:
                    output, h_res = my_mental.forward(p, n_entry, h_1)
                    h = h_res

                sig = torch.nn.Sigmoid()
                output = sig(output)

                out = output.squeeze_(1)
                for i in range(0, batch_sz):
                    preds.append(out[i].detach())

                label = label.squeeze_(1)
                for i in range(0, len(label)):
                    conds.append(label[i].item())

 
    preds = np.array(preds)
    print(preds)
    conds = np.array(conds)
    print(conds)

    np.save('/home/ggreiner/MENTAL/MENTAL_EC_EO_IMPUTED_MDD_PREDICTIONS', preds)
    np.save('/home/ggreiner/MENTAL/MENTAL_EC_EO_IMPUTED_MDD_CONDITIONS', conds)


# running code

epoch = [5]
batches = [5]

learn = 1e-4
weight_decay = 1e-6

for i in range(0, len(epoch)):
    for j in range(0, len(batches)):
        run_train_EO(learn_rate=learn, wd=weight_decay, batch_sz=batches[j], epochs=epoch[i], 
                  outfile="tester")
        