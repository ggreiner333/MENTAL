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
from Model.mentalModel import MENTAL
from Model.mental import MENTAL_EEG


##################################################################################################
##################################################################################################
##################################################################################################

def run_train_EC(learn_rate, wd, batch_sz, epochs, outfile):
    diagnoses = ['-1', 'HEALTHY', 'MDD', 'ADHD', 'SMC', 'OCD', 'TINNITUS', 'INSOMNIA', 'PARKINSON', 'DYSLEXIA',
                'ANXIETY', 'PAIN', 'CHRONIC PAIN', 'PDD NOS', 'BURNOUT', 'BIPOLAR', 'ASPERGER', 
                'DEPERSONALIZATION', 'ASD', 'WHIPLASH', 'MIGRAINE', 'EPILEPSY', 'GTS', 'PANIC', 
                'STROKE', 'TBI', 'ANOREXIA', 'CONVERSION DX', 'DPS', 'DYSPRAXIA', 'LYME', 'MSA-C', 
                'PTSD', 'TRAUMA', 'TUMOR', 'DYSCALCULIA']


    main_dataset = SplitDataset('small_complete_samples_EC_depression.npy', '/data/zhanglab/ggreiner/MENTAL/TDBRAIN')

    splits = []
    if(batch_sz != 15):
        splits = [560, 140, 5]
    else:
        splits = [555, 150]

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

        sensitivity = TP/P
        sens.append(sensitivity)

        specificity = TN/N
        spec.append(specificity)

        '''
        plt.figure(figsize=(15,10))
        plt.hist(vals, bins=np.arange(0, 1.01, 0.05))
        plt.xticks(np.arange(0, 1.01, 0.05))
        plt.yticks(np.arange(0,151,5))
        plt.title("Histogram of Output Values for epoch " + str(epoch))
        plt.ylabel("Count")
        plt.xlabel("Output Value")
        plt.savefig("epoch"+str(epoch)+"_b15_w6_l3_values", pad_inches=0.1)

        print(cvals.__len__()+fvals.__len__())

        plt.figure(figsize=(15,10))
        plt.hist([cvals, fvals], bins=np.arange(0, 1.01, 0.05), label=['Correct', 'Incorrect'], color=['g', 'r'])
        plt.xticks(np.arange(0, 1.01, 0.05))
        plt.yticks(np.arange(0,151,5))
        plt.title("Histogram of Correct and Incorrect Values for epoch " + str(epoch))
        plt.ylabel("Count")
        plt.xlabel("Output Value")
        plt.savefig("epoch"+str(epoch)+"_b15_w6_l3_accvalues", pad_inches=0.1)

        plt.close('all')
        '''

            
    accs = np.array(accs)
    sens = np.array(sens)
    spec = np.array(spec)

    np.save('/home/ggreiner/MENTAL/MENTAL_EC_ACCS', accs)
    np.save('/home/ggreiner/MENTAL/MENTAL_EC_SENS', sens)
    np.save('/home/ggreiner/MENTAL/MENTAL_EC_SPEC', spec)

    labels = np.arange(0, epochs, 1)

    plt.figure(figsize=(15,10))
    plt.plot(labels, accs)
    plt.title("Accuracy of EC MENTAL for " + str(epoch+1) + " epochs, batch size " + str(batch_sz))
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.yticks(ticks=np.arange(0,1.01,0.1))

    plt.savefig("mental_epoch1000_b15_w6_l3_accuracy_ec")
    plt.clf()

    plt.figure(figsize=(15,10))
    plt.plot(labels, sens)
    plt.title("Sensitivity of EC MENTAL for " + str(epoch+1) + " epochs, batch size " + str(batch_sz))
    plt.ylabel("Sensitivity")
    plt.xlabel("Epoch")
    plt.yticks(ticks=np.arange(0,1.01,0.1))

    plt.savefig("mental_epoch1000_b15_w6_l3_sensitivity_ec")
    plt.clf()

    plt.figure(figsize=(15,10))
    plt.plot(labels, spec)
    plt.title("Specificity of EC MENTAL for " + str(epoch+1) + " epochs, batch size " + str(batch_sz))
    plt.ylabel("Specificity")
    plt.xlabel("Epoch")
    plt.yticks(ticks=np.arange(0,1.01,0.1))

    plt.savefig("mental_epoch1000_b15_w6_l3_specificity_ec")
    plt.clf()


def run_train_EO(learn_rate, wd, batch_sz, epochs, outfile):
    diagnoses = ['-1', 'HEALTHY', 'MDD', 'ADHD', 'SMC', 'OCD', 'TINNITUS', 'INSOMNIA', 'PARKINSON', 'DYSLEXIA',
                'ANXIETY', 'PAIN', 'CHRONIC PAIN', 'PDD NOS', 'BURNOUT', 'BIPOLAR', 'ASPERGER', 
                'DEPERSONALIZATION', 'ASD', 'WHIPLASH', 'MIGRAINE', 'EPILEPSY', 'GTS', 'PANIC', 
                'STROKE', 'TBI', 'ANOREXIA', 'CONVERSION DX', 'DPS', 'DYSPRAXIA', 'LYME', 'MSA-C', 
                'PTSD', 'TRAUMA', 'TUMOR', 'DYSCALCULIA']


    main_dataset = SplitDataset('small_complete_samples_EO_depression.npy', '/data/zhanglab/ggreiner/MENTAL/TDBRAIN')

    splits = []
    if(batch_sz != 15):
        splits = [560, 140, 5]
    else:
        splits = [555, 150]

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

        sensitivity = TP/P
        sens.append(sensitivity)

        specificity = TN/N
        spec.append(specificity)
            
    accs = np.array(accs)
    sens = np.array(sens)
    spec = np.array(spec)

    np.save('/home/ggreiner/MENTAL/MENTAL_EO_ACCS', accs)
    np.save('/home/ggreiner/MENTAL/MENTAL_EO_SENS', sens)
    np.save('/home/ggreiner/MENTAL/MENTAL_EO_SPEC', spec)

    labels = np.arange(0, epochs, 1)

    plt.figure(figsize=(15,10))
    plt.plot(labels, accs)
    plt.title("Accuracy of EO MENTAL for " + str(epoch+1) + " epochs, batch size " + str(batch_sz))
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.yticks(ticks=np.arange(0,1.01,0.1))

    plt.savefig("mental_epoch1000_b15_w6_l3_accuracy_eo")
    plt.clf()

    plt.figure(figsize=(15,10))
    plt.plot(labels, sens)
    plt.title("Sensitivity of EO MENTAL for " + str(epoch+1) + " epochs, batch size " + str(batch_sz))
    plt.ylabel("Sensitivity")
    plt.xlabel("Epoch")
    plt.yticks(ticks=np.arange(0,1.01,0.1))

    plt.savefig("mental_epoch1000_b15_w6_l3_sensitivity_eo")
    plt.clf()

    plt.figure(figsize=(15,10))
    plt.plot(labels, spec)
    plt.title("Specificity of EO MENTAL for " + str(epoch+1) + " epochs, batch size " + str(batch_sz))
    plt.ylabel("Specificity")
    plt.xlabel("Epoch")
    plt.yticks(ticks=np.arange(0,1.01,0.1))

    plt.savefig("mental_epoch1000_b15_w6_l3_specificity_eo")
    plt.clf()


def run_train_both(learn_rate, wd, batch_sz, epochs, outfile):
    diagnoses = ['-1', 'HEALTHY', 'MDD', 'ADHD', 'SMC', 'OCD', 'TINNITUS', 'INSOMNIA', 'PARKINSON', 'DYSLEXIA',
                'ANXIETY', 'PAIN', 'CHRONIC PAIN', 'PDD NOS', 'BURNOUT', 'BIPOLAR', 'ASPERGER', 
                'DEPERSONALIZATION', 'ASD', 'WHIPLASH', 'MIGRAINE', 'EPILEPSY', 'GTS', 'PANIC', 
                'STROKE', 'TBI', 'ANOREXIA', 'CONVERSION DX', 'DPS', 'DYSPRAXIA', 'LYME', 'MSA-C', 
                'PTSD', 'TRAUMA', 'TUMOR', 'DYSCALCULIA']


    main_dataset = BSplitDataset('small_complete_samples_EC_EO_depression.npy', '/data/zhanglab/ggreiner/MENTAL/TDBRAIN')

    splits = []
    if(batch_sz != 15):
        splits = [560, 140, 4]
    else:
        splits = [560, 140, 4]

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

        sensitivity = TP/P
        sens.append(sensitivity)

        specificity = TN/N
        spec.append(specificity)
            
    accs = np.array(accs)
    sens = np.array(sens)
    spec = np.array(spec)

    np.save('/home/ggreiner/MENTAL/MENTAL_EC_EO_ACCS', accs)
    np.save('/home/ggreiner/MENTAL/MENTAL_EC_EO_SENS', sens)
    np.save('/home/ggreiner/MENTAL/MENTAL_EC_EO_SPEC', spec)

    labels = np.arange(0, epochs, 1)

    plt.figure(figsize=(15,10))
    plt.plot(labels, accs)
    plt.title("Accuracy of EC+EO MENTAL for " + str(epoch+1) + " epochs, batch size " + str(batch_sz))
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.yticks(ticks=np.arange(0,1.01,0.1))

    plt.savefig("mental_epoch1000_b15_w6_l3_accuracy_ec_eo")
    plt.clf()

    plt.figure(figsize=(15,10))
    plt.plot(labels, sens)
    plt.title("Sensitivity of EC+EO MENTAL for " + str(epoch+1) + " epochs, batch size " + str(batch_sz))
    plt.ylabel("Sensitivity")
    plt.xlabel("Epoch")
    plt.yticks(ticks=np.arange(0,1.01,0.1))

    plt.savefig("mental_epoch1000_b15_w6_l3_sensitivity_ec_eo")
    plt.clf()

    plt.figure(figsize=(15,10))
    plt.plot(labels, spec)
    plt.title("Specificity of EC+EO MENTAL for " + str(epoch+1) + " epochs, batch size " + str(batch_sz))
    plt.ylabel("Specificity")
    plt.xlabel("Epoch")
    plt.yticks(ticks=np.arange(0,1.01,0.1))

    plt.savefig("mental_epoch1000_b15_w6_l3_specificity_ec_eo")
    plt.clf()



# running code

epoch = [1000]
batches = [15]

learn = 1e-3
weight_decay = 1e-6

for i in range(0, len(epoch)):
    for j in range(0, len(batches)):
        run_train_both(learn_rate=learn, wd=weight_decay, batch_sz=batches[j], epochs=epoch[i], 
                  outfile="epoch1000_b15_w6_l3")
        