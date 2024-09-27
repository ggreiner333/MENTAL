import os
from os.path import dirname
from pathlib import Path

import numpy as np
import torch
from torch.utils import data

import mne
import math

import matplotlib.pyplot as plt

from Model.dataset import EEGDataset
from Model.dataset import EEGBothDataset
from Model.mentalModel import MENTAL
from Model.mental import MENTAL_EEG


##################################################################################################
##################################################################################################
##################################################################################################

def run_train_ec(learn_rate, wd, batch_sz, epochs, outfile):


    main_dataset = EEGDataset('only_EC_mdd_healthy_samples.npy', '/data/zhanglab/ggreiner/MENTAL/TDBRAIN')

    splits = [60, 30]
    #if(batch_sz == 15):
    #    splits = [720, 180, 11]
    #else:
    #    splits = [555, 150]

    res = data.random_split(main_dataset, splits)

    train_loader = data.DataLoader(res[0], batch_size=batch_sz, shuffle=True)
    test_loader  = data.DataLoader(res[1], batch_size=batch_sz, shuffle=True)

    my_mental = MENTAL_EEG(130, 30, 1, batch_sz)

    optimizer = torch.optim.Adam(my_mental.parameters(), lr=learn_rate, weight_decay=wd)

    torch.autograd.set_detect_anomaly(True)

    preds = []
    conds = []

    for epoch in range(epochs):
        for (h_entry, p_entry, label) in train_loader:
            label_reshaped = np.reshape(label, (batch_sz,1,1))

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
            psd_final = torch.squeeze(psd_tensor)
            
            h_1 = torch.zeros([2, 1, 30], dtype=torch.float32)

            for p in psd_final:
                output, h_res = my_mental.forward(p, h_1)
                h = h_res

            loss = torch.nn.BCELoss()
            res = loss(output, label_reshaped)

            optimizer.zero_grad()
            res.backward()
            optimizer.step()
        
        
        if(epoch==epochs-1):
            for (h_entry, p_entry, label) in test_loader:

                label_reshaped = np.reshape(label, (batch_sz,1,1))

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
                psd_final = torch.squeeze(psd_tensor)
                
                h_1 = torch.zeros([2, 1, 30], dtype=torch.float32)
                
                for p in psd_final:
                    output, h_res = my_mental.forward(p, h_1)
                    h = h_res

                sig = torch.nn.Sigmoid()
                output = sig(output)

                out = output.squeeze_(1)
                for i in range(0, batch_sz):
                    preds.append(out[i][0].detach())

                label = label.squeeze_(1)
                for i in range(0, len(label)):
                    conds.append(label[i].item())

    preds = np.array(preds)
    print(preds)
    conds = np.array(conds)
    print(conds)

    np.save('/home/ggreiner/MENTAL/outs2/EC_ONLY_MDD_HEALTHY_PREDICTIONS', preds)
    np.save('/home/ggreiner/MENTAL/outs2/EC_ONLY_MDD_HEALTHY_CONDITIONS', conds)


def run_EC():
    # running code

    epoch = [100]
    batches = [5]

    learn = 1e-3
    weight_decay = 1e-6

    for i in range(0, len(epoch)):
        for j in range(0, len(batches)):
            run_train_ec(learn_rate=learn, wd=weight_decay, batch_sz=batches[j], epochs=epoch[i], 
                    outfile="ec_epoch1000_b15_w6_l3")

#run_EC()

def run_train_eo(learn_rate, wd, batch_sz, epochs, outfile):

    main_dataset = EEGDataset('only_EO_mdd_healthy_samples.npy', '/data/zhanglab/ggreiner/MENTAL/TDBRAIN')

    splits = [60, 30]
    #if(batch_sz == 15):
    #    splits = [720, 180, 11]
    #else:
    #    splits = [555, 150]

    res = data.random_split(main_dataset, splits)

    train_loader = data.DataLoader(res[0], batch_size=batch_sz, shuffle=True)
    test_loader  = data.DataLoader(res[1], batch_size=batch_sz, shuffle=True)

    my_mental = MENTAL_EEG(130, 30, 1, batch_sz)

    optimizer = torch.optim.Adam(my_mental.parameters(), lr=learn_rate, weight_decay=wd)

    torch.autograd.set_detect_anomaly(True)

    preds = []
    conds = []

    for epoch in range(epochs):

        for (h_entry, p_entry, label) in train_loader:
            label_reshaped = np.reshape(label, (batch_sz,1,1))

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
            psd_final = torch.squeeze(psd_tensor)
            
            h_1 = torch.zeros([2, 1, 30], dtype=torch.float32)

            for p in psd_final:
                output, h_res = my_mental.forward(p, h_1)
                h = h_res

            loss = torch.nn.BCELoss()
            res = loss(output, label_reshaped)

            optimizer.zero_grad()
            res.backward()
            optimizer.step()
        
        
        if(epoch==epochs-1):
            for (h_entry, p_entry, label) in test_loader:

                label_reshaped = np.reshape(label, (batch_sz,1,1))

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
                psd_final = torch.squeeze(psd_tensor)
                
                h_1 = torch.zeros([2, 1, 30], dtype=torch.float32)
                
                for p in psd_final:
                    output, h_res = my_mental.forward(p, h_1)
                    h = h_res

                sig = torch.nn.Sigmoid()
                output = sig(output)

                out = output.squeeze_(1)
                for i in range(0, batch_sz):
                    preds.append(out[i][0].detach())

                label = label.squeeze_(1)
                for i in range(0, len(label)):
                    conds.append(label[i].item())

    preds = np.array(preds)
    print(preds)
    conds = np.array(conds)
    print(conds)

    np.save('/home/ggreiner/MENTAL/outs2/EO_ONLY_MDD_HEALTHY_PREDICTIONS', preds)
    np.save('/home/ggreiner/MENTAL/outs2/EO_ONLY_MDD_HEALTHY_CONDITIONS', conds)

def run_EO():
    # running code

    epoch = [5]
    batches = [5]

    learn = 1e-3
    weight_decay = 1e-6

    for i in range(0, len(epoch)):
        for j in range(0, len(batches)):
            run_train_eo(learn_rate=learn, wd=weight_decay, batch_sz=batches[j], epochs=epoch[i], 
                    outfile="eo_epoch1000_b15_w6_l3")

#run_EO()

def run_train_both(learn_rate, wd, batch_sz, epochs, outfile):

    main_dataset = EEGBothDataset('only_EC_EO_mdd_healthy_samples.npy', '/data/zhanglab/ggreiner/MENTAL/TDBRAIN')

    splits = [60, 30]
    #if(batch_sz == 15):
    #    splits = [720, 180, 9]
    #else:
    #    splits = [555, 150]

    res = data.random_split(main_dataset, splits)

    train_loader = data.DataLoader(res[0], batch_size=batch_sz, shuffle=True)
    test_loader  = data.DataLoader(res[1], batch_size=batch_sz, shuffle=True)

    my_mental = MENTAL_EEG(130, 30, 1, batch_sz)

    optimizer = torch.optim.Adam(my_mental.parameters(), lr=learn_rate, weight_decay=wd)

    torch.autograd.set_detect_anomaly(True)

    preds = []
    conds = []

    for epoch in range(epochs):

        for (h_entry, p_entry, label) in train_loader:

            label_reshaped = np.reshape(label, (batch_sz,1,1))

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
            psd_final = torch.squeeze(psd_tensor)

            h_1 = torch.zeros([2, 1, 30], dtype=torch.float32)

            for p in psd_final:
                output, h_res = my_mental.forward(p, h_1)
                h = h_res

            loss = torch.nn.BCELoss()
            res = loss(output, label_reshaped)

            optimizer.zero_grad()
            res.backward()
            optimizer.step()
        
        
        if(epoch==epochs-1):
            for (h_entry, p_entry, label) in test_loader:

                label_reshaped = np.reshape(label, (batch_sz,1,1))

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
                psd_final = torch.squeeze(psd_tensor)
                
                h_1 = torch.zeros([2, 1, 30], dtype=torch.float32)
                
                for p in psd_final:
                    output, h_res = my_mental.forward(p, h_1)
                    h = h_res

                sig = torch.nn.Sigmoid()
                output = sig(output)

                out = output.squeeze_(1)
                for i in range(0, batch_sz):
                    preds.append(out[i][0].detach())

                label = label.squeeze_(1)
                for i in range(0, len(label)):
                    conds.append(label[i].item())

    preds = np.array(preds)
    print(preds)
    conds = np.array(conds)
    print(conds)

    np.save('/home/ggreiner/MENTAL/outs2/EC_EO_ONLY_MDD_HEALTHY_PREDICTIONS', preds)
    np.save('/home/ggreiner/MENTAL/outs2/EC_EO_ONLY_MDD_HEALTHY_CONDITIONS', conds)

def run_EC_EO():
    # running code

    epoch = [5]
    batches = [5]

    learn = 1e-3
    weight_decay = 1e-6

    for i in range(0, len(epoch)):
        for j in range(0, len(batches)):
            run_train_both(learn_rate=learn, wd=weight_decay, batch_sz=batches[j], epochs=epoch[i], 
                    outfile="ec_eo_epoch1000_b15_w6_l3")

run_EC_EO()