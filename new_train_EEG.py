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

    accs = []
    sens = []
    spec = []

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
        
        
        correct = 0
        vals = []
        cvals = []
        fvals = []
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

            
    accs = np.array(accs)
    sens = np.array(sens)
    spec = np.array(spec)

    np.save('/home/ggreiner/MENTAL/EC_ONLY_MDD_HEALTHY_ACCS', accs)
    np.save('/home/ggreiner/MENTAL/EC_ONLY_MDD_HEALTHY_SENS', sens)
    np.save('/home/ggreiner/MENTAL/EC_ONLY_MDD_HEALTHY_SPEC', spec)

    labels = np.arange(0, epochs, 1)

    plt.figure(figsize=(15,10))
    plt.plot(labels, accs)
    plt.title("EC Accuracy of MDD v HEALTHY Model for " + str(epoch) + " epochs, batch size " + str(batch_sz))
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.yticks(ticks=np.arange(0,1.01,0.1))

    plt.savefig("ec_mdd_healthy_epoch1000_b15_w6_l3_accuracy")
    plt.clf()

    plt.figure(figsize=(15,10))
    plt.plot(labels, sens)
    plt.title("EC Sensitivity of MDD v HEALTHY Model for " + str(epoch) + " epochs, batch size " + str(batch_sz))
    plt.ylabel("Sensitivity")
    plt.xlabel("Epoch")
    plt.yticks(ticks=np.arange(0,1.01,0.1))

    plt.savefig("ec_mdd_healthy_epoch1000_b15_w6_l3_sensitivity")
    plt.clf()

    plt.figure(figsize=(15,10))
    plt.plot(labels, spec)
    plt.title("EC Specificity of MDD v HEALTHY Model for " + str(epoch) + " epochs, batch size " + str(batch_sz))
    plt.ylabel("Specificity")
    plt.xlabel("Epoch")
    plt.yticks(ticks=np.arange(0,1.01,0.1))

    plt.savefig("ec_mdd_healthy_epoch1000_b15_w6_l3_specificity")
    plt.clf()

def run_EC():
    # running code

    epoch = [1000]
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

    accs = []
    sens = []
    spec = []

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
        
        
        correct = 0
        vals = []
        cvals = []
        fvals = []
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

            
    accs = np.array(accs)
    sens = np.array(sens)
    spec = np.array(spec)

    np.save('/home/ggreiner/MENTAL/EO_ONLY_MDD_HEALTHY_ACCS', accs)
    np.save('/home/ggreiner/MENTAL/EO_ONLY_MDD_HEALTHY_SENS', sens)
    np.save('/home/ggreiner/MENTAL/EO_ONLY_MDD_HEALTHY_SPEC', spec)

    labels = np.arange(0, epochs, 1)

    plt.figure(figsize=(15,10))
    plt.plot(labels, accs)
    plt.title("EO Accuracy of MDD v HEALTHY Model for " + str(epoch) + " epochs, batch size " + str(batch_sz))
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.yticks(ticks=np.arange(0,1.01,0.1))

    plt.savefig("eo_mdd_healthy_epoch1000_b15_w6_l3_accuracy")
    plt.clf()

    plt.figure(figsize=(15,10))
    plt.plot(labels, sens)
    plt.title("EO Sensitivity of MDD v HEALTHY Model for " + str(epoch) + " epochs, batch size " + str(batch_sz))
    plt.ylabel("Sensitivity")
    plt.xlabel("Epoch")
    plt.yticks(ticks=np.arange(0,1.01,0.1))

    plt.savefig("eo_mdd_healthy_epoch1000_b15_w6_l3_sensitivity")
    plt.clf()

    plt.figure(figsize=(15,10))
    plt.plot(labels, spec)
    plt.title("EO Specificity of MDD v HEALTHY Model for " + str(epoch) + " epochs, batch size " + str(batch_sz))
    plt.ylabel("Specificity")
    plt.xlabel("Epoch")
    plt.yticks(ticks=np.arange(0,1.01,0.1))

    plt.savefig("eo_mdd_healthy_epoch1000_b15_w6_l3_specificity")
    plt.clf()

def run_EO():
    # running code

    epoch = [1000]
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

    accs = []
    sens = []
    spec = []

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
        
        
        correct = 0
        vals = []
        cvals = []
        fvals = []
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

        if(epoch==100):
            c_accs = np.array(accs)
            c_sens = np.array(sens)
            c_spec = np.array(spec)

            np.save('/home/ggreiner/MENTAL/EO_EC_MDD_HEALTHY_ACCS_e100', c_accs)
            np.save('/home/ggreiner/MENTAL/EO_EC_MDD_HEALTHY_SENS_e100', c_sens)
            np.save('/home/ggreiner/MENTAL/EO_EC_MDD_HEALTHY_SPEC_e100', c_spec)
            
    accs = np.array(accs)
    sens = np.array(sens)
    spec = np.array(spec)

    np.save('/home/ggreiner/MENTAL/EO_EC_MDD_HEALTHY_ACCS', accs)
    np.save('/home/ggreiner/MENTAL/EO_EC_MDD_HEALTHY_SENS', sens)
    np.save('/home/ggreiner/MENTAL/EO_EC_MDD_HEALTHY_SPEC', spec)

    labels = np.arange(0, epochs, 1)

    plt.figure(figsize=(15,10))
    plt.plot(labels, accs)
    plt.title("EC+EO Accuracy of MDD v HEALTHY Model for " + str(epoch) + " epochs, batch size " + str(batch_sz))
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.yticks(ticks=np.arange(0,1.01,0.1))

    plt.savefig("ec_eo_mdd_healthy_epoch1000_b15_w6_l3_accuracy")
    plt.clf()

    plt.figure(figsize=(15,10))
    plt.plot(labels, sens)
    plt.title("EC+EO Sensitivity of MDD v HEALTHY Model for " + str(epoch) + " epochs, batch size " + str(batch_sz))
    plt.ylabel("Sensitivity")
    plt.xlabel("Epoch")
    plt.yticks(ticks=np.arange(0,1.01,0.1))

    plt.savefig("ec_eo_mdd_healthy_epoch1000_b15_w6_l3_sensitivity")
    plt.clf()

    plt.figure(figsize=(15,10))
    plt.plot(labels, spec)
    plt.title("EC+EO Specificity of MDD v HEALTHY Model for " + str(epoch) + " epochs, batch size " + str(batch_sz))
    plt.ylabel("Specificity")
    plt.xlabel("Epoch")
    plt.yticks(ticks=np.arange(0,1.01,0.1))

    plt.savefig("ec_eo_mdd_healthy_epoch1000_b15_w6_l3_specificity")
    plt.clf()    

def run_EC_EO():
    # running code

    epoch = [1000]
    batches = [5]

    learn = 1e-3
    weight_decay = 1e-6

    for i in range(0, len(epoch)):
        for j in range(0, len(batches)):
            run_train_both(learn_rate=learn, wd=weight_decay, batch_sz=batches[j], epochs=epoch[i], 
                    outfile="ec_eo_epoch1000_b15_w6_l3")

run_EC_EO()