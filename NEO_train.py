import os
from os.path import dirname
from pathlib import Path

import numpy as np
import torch
from torch.utils import data

import mne
import math

import matplotlib.pyplot as plt

from Model.dataset import NEODataset
from Model.neo_model import NEO_NN



##################################################################################################
##################################################################################################
##################################################################################################

def run_train_neo(learn_rate, batch_sz, epochs):
    diagnoses = ['-1', 'HEALTHY', 'MDD', 'ADHD', 'SMC', 'OCD', 'TINNITUS', 'INSOMNIA', 'PARKINSON', 'DYSLEXIA',
                'ANXIETY', 'PAIN', 'CHRONIC PAIN', 'PDD NOS', 'BURNOUT', 'BIPOLAR', 'ASPERGER', 
                'DEPERSONALIZATION', 'ASD', 'WHIPLASH', 'MIGRAINE', 'EPILEPSY', 'GTS', 'PANIC', 
                'STROKE', 'TBI', 'ANOREXIA', 'CONVERSION DX', 'DPS', 'DYSPRAXIA', 'LYME', 'MSA-C', 
                'PTSD', 'TRAUMA', 'TUMOR', 'DYSCALCULIA']


    main_dataset = NEODataset('only_NEO_samples.npy', '/data/zhanglab/ggreiner/MENTAL/TDBRAIN')

    splits = []
    if(batch_sz == 15):
        splits = [540, 135, 3]
    else:
        splits = [555, 150]

    res = data.random_split(main_dataset, splits)

    train_loader = data.DataLoader(res[0], batch_size=batch_sz, shuffle=True)
    test_loader  = data.DataLoader(res[1], batch_size=batch_sz, shuffle=True)

    my_mental = NEO_NN(60, 1, batch_sz)

    optimizer = torch.optim.Adam(my_mental.parameters(), lr=learn_rate)

    torch.autograd.set_detect_anomaly(True)

    accs = []
    sens = []
    spec = []

    for epoch in range(epochs):
        for (n_entry, label) in train_loader:
            out = my_mental.forward(n_entry)

            loss = torch.nn.MSELoss()
            res = loss(out, label)

            optimizer.zero_grad()
            res.backward()
            optimizer.step()
        
        
        correct = 0
        vals = []
        cvals = []
        fvals = []
        for (n_entry, label) in test_loader:
            out = my_mental.forward(n_entry)
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

    labels = np.arange(0, epochs, 1)

    plt.figure(figsize=(15,10))
    plt.plot(labels, accs)
    plt.title("NEO Accuracy of Model for " + str(epoch) + " epochs, batch size " + str(batch_sz))
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.yticks(ticks=np.arange(0,1.01,0.1))

    plt.savefig("NEO_epoch10_b15_l3_accuracy")
    plt.clf()

    plt.figure(figsize=(15,10))
    plt.plot(labels, sens)
    plt.title("NEO Sensitivity of Model for " + str(epoch) + " epochs, batch size " + str(batch_sz))
    plt.ylabel("Sensitivity")
    plt.xlabel("Epoch")
    plt.yticks(ticks=np.arange(0,1.01,0.1))

    plt.savefig("NEO_epoch10_b15_l3_sensitivity")
    plt.clf()

    plt.figure(figsize=(15,10))
    plt.plot(labels, spec)
    plt.title("NEO Specificity of Model for " + str(epoch) + " epochs, batch size " + str(batch_sz))
    plt.ylabel("Specificity")
    plt.xlabel("Epoch")
    plt.yticks(ticks=np.arange(0,1.01,0.1))

    plt.savefig("NEO_epoch10_b15_l3_specificity")
    plt.clf()

def run_NEO():
    # running code

    epoch = 10
    batch = 15

    learn = 1e-3
    
    run_train_neo(learn_rate=learn, batch_sz=batch, epochs=epoch)

run_NEO()

