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

    main_dataset = SplitDataset('normalized_small_imputed_complete_samples_EC_depression.npy', '/data/zhanglab/ggreiner/MENTAL/TDBRAIN')

    splits = []
    splits = [735,225, 8]
    """
    if(batch_sz != 15):
        splits = [560, 140, 5]
    else:
        splits = [555, 150]
    """

    res = data.random_split(main_dataset, splits)

    train_loader = data.DataLoader(res[0], batch_size=batch_sz, shuffle=True)
    test_loader  = data.DataLoader(res[1], batch_size=batch_sz, shuffle=True)

    my_mental = MENTAL(60, 30, 1, batch_sz)

    optimizer = torch.optim.Adam(my_mental.parameters(), lr=learn_rate)

    torch.autograd.set_detect_anomaly(True)

    accs = []
    sens = []
    spec = []

    for epoch in range(epochs):
        
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
        
        
        preds = []
        conds = []
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

            print(output)
            out = output.squeeze_(1)
            for i in range(0, batch_sz):
                for j in range(0, len(out[i])):
                    preds.append(out[i][j].detach())
                    print(preds)

            label = label.squeeze_(1)
            for i in range(0, len(label)):
                conds.append(label[i].item())

 
    preds = np.array(preds)
    print(preds)
    conds = np.array(conds)
    print(conds)

    np.save('/home/ggreiner/MENTAL/MENTAL_EC_IMPUTED_MDD_PREDICTIONS', preds)
    np.save('/home/ggreiner/MENTAL/MENTAL_EC_IMPUTED_MDD_CONDITIONS', conds)


def run_train_EO(learn_rate, wd, batch_sz, epochs, outfile):

    main_dataset = SplitDataset('normalized_small_imputed_complete_samples_EO_depression.npy', '/data/zhanglab/ggreiner/MENTAL/TDBRAIN')

    splits = []
    splits = [735,225, 8]
    """
    if(batch_sz != 15):
        splits = [560, 140, 5]
    else:
        splits = [555, 150]
    """

    res = data.random_split(main_dataset, splits)

    train_loader = data.DataLoader(res[0], batch_size=batch_sz, shuffle=True)
    test_loader  = data.DataLoader(res[1], batch_size=batch_sz, shuffle=True)

    my_mental = MENTAL(60, 30, 1, batch_sz)


    optimizer = torch.optim.Adam(my_mental.parameters(), lr=learn_rate)

    torch.autograd.set_detect_anomaly(True)

    accs = []
    sens = []
    spec = []

    for epoch in range(epochs):
        
        for (h_entry, n_entry, p_entry, label) in train_loader:

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
            
            h_1 = torch.zeros([2, batch_sz, 30], dtype=torch.float32)

            for p in psd_tensor:
                output, h_res = my_mental.forward(p, n_entry, h_1)
                h = h_res

            loss = torch.nn.BCELoss()
            res = loss(output, label_reshaped)

            optimizer.zero_grad()
            res.backward()
            optimizer.step()
        
        
        correct = 0
        vals = []
        cvals = []
        fvals = []
        for (h_entry, n_entry, p_entry, label) in test_loader:

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
            
            h_1 = torch.zeros([2, batch_sz, 30], dtype=torch.float32)
            
            for p in psd_tensor:
                output, h_res = my_mental.forward(p, n_entry, h_1)
                h = h_res

            out = output.squeeze_(1)
            preds = []
            for i in range(0, batch_sz):
                for j in range(0, len(out[i])):
                    if(out[i][j] >= 0.5):
                        preds.append(1)
                    else:
                        preds.append(0)
                    vals.append(out[i][j].detach())

            label = label.squeeze_(1)
            conds = []

            for i in range(0, len(label)):
                conds.append(label[i].item())

            # Variables for calculating specificity and sensitivity
            N = 0
            P = 0
            TP = 0
            TN = 0
            for i in range(0, len(conds)):
                lb = conds[i]
                pd = preds[i]
                if(lb == 1):
                    P+=1
                    if(lb==pd): 
                        correct += 1
                        TP+=1
                        cvals.append(vals[i])
                    else:
                        fvals.append(vals[i])
                if(lb == 0):
                    N+=1
                    if(lb==pd):
                        correct += 1
                        TN+=1
                        cvals.append(vals[i])
                    else:
                        fvals.append(vals[i])

            
        total = (test_loader.__len__())*batch_sz
        acc = correct/total
        accs.append(acc)
        print(acc)

        sensitivity = 1 if(P == 0) else TP/P
        sens.append(sensitivity)

        specificity = 1 if(N==0) else TN/N
        spec.append(specificity)

        if(epoch%100==0):
            np.save('/home/ggreiner/MENTAL/MENTAL_EO_IMPUTED_MDD_ACCS'+str(epoch), accs)
        
    accs = np.array(accs)
    sens = np.array(sens)
    spec = np.array(spec)

    np.save('/home/ggreiner/MENTAL/MENTAL_EO_IMPUTED_MDD_ACCS', accs)
    np.save('/home/ggreiner/MENTAL/MENTAL_EO_IMPUTED_MDD_SENS', sens)
    np.save('/home/ggreiner/MENTAL/MENTAL_EO_IMPUTED_MDD_SPEC', spec)

    """
    labels = np.arange(0, epochs, 1)

    plt.figure(figsize=(15,10))
    plt.plot(labels, accs)
    plt.title("Accuracy of BCE ADHD EO MENTAL for " + str(epoch+1) + " epochs, batch size " + str(batch_sz))
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.yticks(ticks=np.arange(0,1.01,0.1))

    plt.savefig("BCE_adhd_mental_epoch1000_b15_w6_l3_accuracy_eo")
    plt.clf()

    plt.figure(figsize=(15,10))
    plt.plot(labels, sens)
    plt.title("Sensitivity of BCE ADHD EO MENTAL for " + str(epoch+1) + " epochs, batch size " + str(batch_sz))
    plt.ylabel("Sensitivity")
    plt.xlabel("Epoch")
    plt.yticks(ticks=np.arange(0,1.01,0.1))

    plt.savefig("BCE_adhd_mental_epoch1000_b15_w6_l3_sensitivity_eo")
    plt.clf()

    plt.figure(figsize=(15,10))
    plt.plot(labels, spec)
    plt.title("Specificity of BCE ADHD EO MENTAL for " + str(epoch+1) + " epochs, batch size " + str(batch_sz))
    plt.ylabel("Specificity")
    plt.xlabel("Epoch")
    plt.yticks(ticks=np.arange(0,1.01,0.1))

    plt.savefig("BCE_adhd_mental_epoch1000_b15_w6_l3_specificity_eo")
    plt.clf()
    """


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

    accs = []
    sens = []
    spec = []

    for epoch in range(epochs):
        
        for (h_entry, n_entry, p_entry, label) in train_loader:

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

            h_1 = torch.zeros([2, batch_sz, 30], dtype=torch.float32)

            for p in psd_tensor:
                output, h_res = my_mental.forward(p, n_entry, h_1)
                h = h_res

            loss = torch.nn.BCELoss()
            res = loss(output, label_reshaped)

            optimizer.zero_grad()
            res.backward()
            optimizer.step()
        
        
        correct = 0
        vals = []
        cvals = []
        fvals = []
        for (h_entry, n_entry, p_entry, label) in test_loader:

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
            
            h_1 = torch.zeros([2, batch_sz, 30], dtype=torch.float32)
            
            for p in psd_tensor:
                output, h_res = my_mental.forward(p, n_entry, h_1)
                h = h_res

            out = output.squeeze_(1)
            preds = []
            for i in range(0, batch_sz):
                for j in range(0, len(out[i])):
                    if(out[i][j] >= 0.5):
                        preds.append(1)
                    else:
                        preds.append(0)
                    vals.append(out[i][j].detach())

            label = label.squeeze_(1)
            conds = []

            for i in range(0, len(label)):
                conds.append(label[i].item())

            # Variables for calculating specificity and sensitivity
            N = 0
            P = 0
            TP = 0
            TN = 0
            for i in range(0, len(conds)):
                lb = conds[i]
                pd = preds[i]
                if(lb == 1):
                    P+=1
                    if(lb==pd): 
                        correct += 1
                        TP+=1
                        cvals.append(vals[i])
                    else:
                        fvals.append(vals[i])
                if(lb == 0):
                    N+=1
                    if(lb==pd):
                        correct += 1
                        TN+=1
                        cvals.append(vals[i])
                    else:
                        fvals.append(vals[i])

            

        total = (test_loader.__len__())*batch_sz
        acc = correct/total
        accs.append(acc)
        print(acc)

        sensitivity = 1 if(P == 0) else TP/P
        sens.append(sensitivity)

        specificity = 1 if(N==0) else TN/N
        spec.append(specificity)

        if(epoch%100==0):
            np.save('/home/ggreiner/MENTAL/MENTAL_EC_EO_IMPUTED_MDD_ACCS'+str(epoch), accs)

       
    accs = np.array(accs)
    sens = np.array(sens)
    spec = np.array(spec)

    np.save('/home/ggreiner/MENTAL/MENTAL_EC_EO_IMPUTED_MDD_ACCS', accs)
    np.save('/home/ggreiner/MENTAL/MENTAL_EC_EO_IMPUTED_MDD_SENS', sens)
    np.save('/home/ggreiner/MENTAL/MENTAL_EC_EO_IMPUTED_MDD_SPEC', spec)

    """
    labels = np.arange(0, epochs, 1)

    plt.figure(figsize=(15,10))
    plt.plot(labels, accs)
    plt.title("Accuracy of ADHD EC+EO MENTAL for " + str(epoch+1) + " epochs, batch size " + str(batch_sz))
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.yticks(ticks=np.arange(0,1.01,0.1))

    plt.savefig("adhd_mental_epoch1000_b15_w6_l3_accuracy_ec_eo")
    plt.clf()

    plt.figure(figsize=(15,10))
    plt.plot(labels, sens)
    plt.title("Sensitivity of ADHD EC+EO MENTAL for " + str(epoch+1) + " epochs, batch size " + str(batch_sz))
    plt.ylabel("Sensitivity")
    plt.xlabel("Epoch")
    plt.yticks(ticks=np.arange(0,1.01,0.1))

    plt.savefig("adhd_mental_epoch1000_b15_w6_l3_sensitivity_ec_eo")
    plt.clf()

    plt.figure(figsize=(15,10))
    plt.plot(labels, spec)
    plt.title("Specificity of ADHD EC+EO MENTAL for " + str(epoch+1) + " epochs, batch size " + str(batch_sz))
    plt.ylabel("Specificity")
    plt.xlabel("Epoch")
    plt.yticks(ticks=np.arange(0,1.01,0.1))

    plt.savefig("adhd_mental_epoch1000_b15_w6_l3_specificity_ec_eo")
    plt.clf()
    """


def run_train_EC_Multi(learn_rate, wd, batch_sz, epochs, outfile):

    main_dataset = MSplitDataset('normalized_small_imputed_complete_samples_EC_top5.npy', '/data/zhanglab/ggreiner/MENTAL/TDBRAIN')

    splits = [615, 165, 11]

    res = data.random_split(main_dataset, splits, generator=torch.Generator().manual_seed(43))

    train_loader = data.DataLoader(res[0], batch_size=batch_sz, shuffle=True)
    test_loader  = data.DataLoader(res[1], batch_size=batch_sz, shuffle=True)

    my_mental = MENTAL(60, 30, 5, batch_sz)

    optimizer = torch.optim.Adam(my_mental.parameters(), lr=learn_rate)

    torch.autograd.set_detect_anomaly(True)

    accs = []
    sens = []
    spec = []
    confusion = [[[0]*5 for i in range(0,5)] for i in range(0, epochs)]

    for epoch in range(epochs):
        
        for (h_entry, n_entry, p_entry, label) in train_loader:

            label = label.type(torch.float32)
            #print(f"lable: {label}")

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

            #output = output.type(torch.float32)
            #print(f"out  : {output}")
            #print(f"shape: {output.shape}")

            loss = torch.nn.CrossEntropyLoss()
            res = loss(output, label)
            #print(f"loss: {res}")

            optimizer.zero_grad()
            res.backward()
            optimizer.step()
        
        
        correct = 0
        vals = []
        cvals = []
        fvals = []
        for (h_entry, n_entry, p_entry, label) in test_loader:

            label = label.type(torch.float32)

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

            #out = output.squeeze_(1)
            soft = torch.nn.Softmax(dim=1)
            output = soft(output)
            
            #print(f"out  : {output}")
            #print(f"shape: {output.shape}")

            preds = []
            for i in range(0, batch_sz):
                mx = 0
                loc = 0
                for j in range(0, 5):
                    if(output[i][j] > mx):
                        mx = output[i][j]
                        loc = j
                #if(i==14):
                #    print(f"row: {output[i][0]}")
                #    print(f"max: {loc}")
                #    print(f"val: {mx}")
                preds.append(loc)

            conds = []
            for i in range(0, batch_sz):
                for j in range(0, 5):
                    if(label[i][j] > 0):
                        conds.append(j)
                        break
            
            for i in range(0, len(conds)):
                lb = conds[i]
                pd = preds[i]
                if(lb==pd): 
                    correct += 1
                confusion[epoch][lb][pd] = confusion[epoch][lb][pd] + 1
            
        total = (test_loader.__len__())*batch_sz
        acc = correct/total
        accs.append(acc)
        print(f"Epoch {epoch}: {acc}")

        if(epoch%750==0):
            np.save('/home/ggreiner/MENTAL/new_TOP5_1e5_MENTAL_EC_IMPUTED_ACCS'+str(epoch), accs)
            np.save('/home/ggreiner/MENTAL/new_TOP5_1e5_MENTAL_EC_IMPUTED_CONFUSION'+str(epoch), confusion)
 
    accs = np.array(accs)
    np.save('/home/ggreiner/MENTAL/new_TOP5_1e5_MENTAL_EC_IMPUTED_ACCS', accs)
    confusion = np.array(confusion)
    np.save('/home/ggreiner/MENTAL/new_TOP5_1e5_MENTAL_EC_IMPUTED_CONFUSION', confusion)

def run_train_EO_Multi(learn_rate, wd, batch_sz, epochs, outfile):

    main_dataset = MSplitDataset('normalized_small_imputed_complete_samples_EO_top5.npy', '/data/zhanglab/ggreiner/MENTAL/TDBRAIN')

    splits = [615, 165, 10]

    res = data.random_split(main_dataset, splits, generator=torch.Generator().manual_seed(43))

    train_loader = data.DataLoader(res[0], batch_size=batch_sz, shuffle=True)
    test_loader  = data.DataLoader(res[1], batch_size=batch_sz, shuffle=True)

    my_mental = MENTAL(60, 30, 5, batch_sz)

    optimizer = torch.optim.Adam(my_mental.parameters(), lr=learn_rate)

    torch.autograd.set_detect_anomaly(True)

    accs = []
    sens = []
    spec = []
    confusion = [[[0]*5 for i in range(0,5)] for i in range(0, epochs)]

    for epoch in range(epochs):
        
        for (h_entry, n_entry, p_entry, label) in train_loader:

            label = label.type(torch.float32)
            #print(f"lable: {label}")

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

            #output = output.type(torch.float32)
            #print(f"out  : {output}")
            #print(f"shape: {output.shape}")

            loss = torch.nn.CrossEntropyLoss()
            res = loss(output, label)
            #print(f"loss: {res}")

            optimizer.zero_grad()
            res.backward()
            optimizer.step()
        
        
        correct = 0
        vals = []
        cvals = []
        fvals = []
        for (h_entry, n_entry, p_entry, label) in test_loader:

            label = label.type(torch.float32)

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

            #out = output.squeeze_(1)
            soft = torch.nn.Softmax(dim=1)
            output = soft(output)
            
            #print(f"out  : {output}")
            #print(f"shape: {output.shape}")

            preds = []
            for i in range(0, batch_sz):
                mx = 0
                loc = 0
                for j in range(0, 5):
                    if(output[i][j] > mx):
                        mx = output[i][j]
                        loc = j
                #if(i==14):
                #    print(f"row: {output[i][0]}")
                #    print(f"max: {loc}")
                #    print(f"val: {mx}")
                preds.append(loc)

            conds = []
            for i in range(0, batch_sz):
                for j in range(0, 5):
                    if(label[i][j] > 0):
                        conds.append(j)
                        break
            
            for i in range(0, len(conds)):
                lb = conds[i]
                pd = preds[i]
                if(lb==pd): 
                    correct += 1
                confusion[epoch][lb][pd] = confusion[epoch][lb][pd] + 1
            
        total = (test_loader.__len__())*batch_sz
        acc = correct/total
        accs.append(acc)
        print(f"Epoch {epoch}: {acc}")

        if(epoch%750==0):
            np.save('/home/ggreiner/MENTAL/new_TOP5_1e5_MENTAL_EO_IMPUTED_ACCS'+str(epoch), accs)
            np.save('/home/ggreiner/MENTAL/new_TOP5_1e5_MENTAL_EO_IMPUTED_CONFUSION'+str(epoch), confusion)
 
    accs = np.array(accs)
    np.save('/home/ggreiner/MENTAL/new_TOP5_1e5_MENTAL_EO_IMPUTED_ACCS', accs)
    confusion = np.array(confusion)
    np.save('/home/ggreiner/MENTAL/new_TOP5_1e5_MENTAL_EO_IMPUTED_CONFUSION', confusion)


def run_train_EC_Multi_top3(learn_rate, wd, batch_sz, epochs, outfile):

    main_dataset = M3SplitDataset('normalized_small_imputed_complete_samples_EC_top3.npy', '/data/zhanglab/ggreiner/MENTAL/TDBRAIN')

    splits = [540, 150, 8]

    res = data.random_split(main_dataset, splits, generator=torch.Generator().manual_seed(40))

    train_loader = data.DataLoader(res[0], batch_size=batch_sz, shuffle=True)
    test_loader  = data.DataLoader(res[1], batch_size=batch_sz, shuffle=True)

    my_mental = MENTAL(60, 30, 3, batch_sz)

    optimizer = torch.optim.Adam(my_mental.parameters(), lr=learn_rate)

    torch.autograd.set_detect_anomaly(True)

    accs = []
    sens = []
    spec = []
    confusion = [[[0]*3 for i in range(0,3)] for i in range(0, epochs)]

    for epoch in range(epochs):
        
        for (h_entry, n_entry, p_entry, label) in train_loader:

            label = label.type(torch.float32)
            #print(f"lable: {label}")

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

            #output = output.type(torch.float32)
            #print(f"out  : {output}")
            #print(f"shape: {output.shape}")

            loss = torch.nn.CrossEntropyLoss()
            res = loss(output, label)
            #print(f"loss: {res}")

            optimizer.zero_grad()
            res.backward()
            optimizer.step()
        
        
        correct = 0
        vals = []
        cvals = []
        fvals = []
        for (h_entry, n_entry, p_entry, label) in test_loader:

            label = label.type(torch.float32)

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

            #out = output.squeeze_(1)
            soft = torch.nn.Softmax(dim=1)
            output = soft(output)
            
            #print(f"out  : {output}")
            #print(f"shape: {output.shape}")

            preds = []
            for i in range(0, batch_sz):
                mx = 0
                loc = 0
                for j in range(0, 3):
                    if(output[i][j] > mx):
                        mx = output[i][j]
                        loc = j
                #if(i==14):
                #    print(f"row: {output[i][0]}")
                #    print(f"max: {loc}")
                #    print(f"val: {mx}")
                preds.append(loc)

            conds = []
            for i in range(0, batch_sz):
                for j in range(0, 3):
                    if(label[i][j] > 0):
                        conds.append(j)
                        break
            
            for i in range(0, len(conds)):
                lb = conds[i]
                pd = preds[i]
                if(lb==pd): 
                    correct += 1
                confusion[epoch][lb][pd] = confusion[epoch][lb][pd] + 1
            
        total = (test_loader.__len__())*batch_sz
        acc = correct/total
        accs.append(acc)
        print(f"Epoch {epoch}: {acc}")

        if(epoch%500==0):
            np.save('/home/ggreiner/MENTAL/TOP3_1e4_MENTAL_EC_IMPUTED_ACCS'+str(epoch), accs)
            np.save('/home/ggreiner/MENTAL/TOP3_1e4_MENTAL_EC_IMPUTED_CONFUSION'+str(epoch), confusion)
 
    accs = np.array(accs)
    np.save('/home/ggreiner/MENTAL/TOP3_1e4_MENTAL_EC_IMPUTED_ACCS', accs)
    confusion = np.array(confusion)
    np.save('/home/ggreiner/MENTAL/TOP3_1e4_MENTAL_EC_IMPUTED_CONFUSION', confusion)

def run_train_EO_Multi_top3(learn_rate, wd, batch_sz, epochs, outfile):

    main_dataset = M3SplitDataset('normalized_small_imputed_complete_samples_EO_top3.npy', '/data/zhanglab/ggreiner/MENTAL/TDBRAIN')

    splits = [540, 150, 7]

    res = data.random_split(main_dataset, splits, generator=torch.Generator().manual_seed(40))

    train_loader = data.DataLoader(res[0], batch_size=batch_sz, shuffle=True)
    test_loader  = data.DataLoader(res[1], batch_size=batch_sz, shuffle=True)

    my_mental = MENTAL(60, 30, 3, batch_sz)

    optimizer = torch.optim.Adam(my_mental.parameters(), lr=learn_rate)

    torch.autograd.set_detect_anomaly(True)

    accs = []
    sens = []
    spec = []
    confusion = [[[0]*3 for i in range(0,3)] for i in range(0, epochs)]

    for epoch in range(epochs):
        
        for (h_entry, n_entry, p_entry, label) in train_loader:

            label = label.type(torch.float32)
            #print(f"lable: {label}")

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

            #output = output.type(torch.float32)
            #print(f"out  : {output}")
            #print(f"shape: {output.shape}")

            loss = torch.nn.CrossEntropyLoss()
            res = loss(output, label)
            #print(f"loss: {res}")

            optimizer.zero_grad()
            res.backward()
            optimizer.step()
        
        
        correct = 0
        vals = []
        cvals = []
        fvals = []
        for (h_entry, n_entry, p_entry, label) in test_loader:

            label = label.type(torch.float32)

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

            #out = output.squeeze_(1)
            soft = torch.nn.Softmax(dim=1)
            output = soft(output)
            
            #print(f"out  : {output}")
            #print(f"shape: {output.shape}")

            preds = []
            for i in range(0, batch_sz):
                mx = 0
                loc = 0
                for j in range(0, 3):
                    if(output[i][j] > mx):
                        mx = output[i][j]
                        loc = j
                #if(i==14):
                #    print(f"row: {output[i][0]}")
                #    print(f"max: {loc}")
                #    print(f"val: {mx}")
                preds.append(loc)

            conds = []
            for i in range(0, batch_sz):
                for j in range(0, 3):
                    if(label[i][j] > 0):
                        conds.append(j)
                        break
            
            for i in range(0, len(conds)):
                lb = conds[i]
                pd = preds[i]
                if(lb==pd): 
                    correct += 1
                confusion[epoch][lb][pd] = confusion[epoch][lb][pd] + 1
            
        total = (test_loader.__len__())*batch_sz
        acc = correct/total
        accs.append(acc)
        print(f"Epoch {epoch}: {acc}")
        #print(confusion)

        if(epoch%500==0):
            np.save('/home/ggreiner/MENTAL/TOP3_1e4_MENTAL_EO_IMPUTED_ACCS'+str(epoch), accs)
            np.save('/home/ggreiner/MENTAL/TOP3_1e4_MENTAL_EO_IMPUTED_CONFUSION'+str(epoch), confusion)
 
    accs = np.array(accs)
    np.save('/home/ggreiner/MENTAL/TOP3_1e4_MENTAL_EO_IMPUTED_ACCS', accs)
    confusion = np.array(confusion)
    np.save('/home/ggreiner/MENTAL/TOP3_1e4_MENTAL_EO_IMPUTED_CONFUSION', confusion)


# running code

epoch = [1]
batches = [15]

learn = 1e-5
weight_decay = 1e-6

for i in range(0, len(epoch)):
    for j in range(0, len(batches)):
        run_train_EC(learn_rate=learn, wd=weight_decay, batch_sz=batches[j], epochs=epoch[i], 
                  outfile="tester")
        