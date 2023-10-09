import os
from os.path import dirname
from pathlib import Path

import numpy as np
import torch
from torch.utils import data

import mne
import math


from Model.dataset import MultiModalDataset
from Model.dataset import SplitDataset
from Model.mentalModel import MENTAL


##################################################################################################
##################################################################################################
##################################################################################################

def run_train(lr, outfile):
    diagnoses = ['-1', 'HEALTHY', 'MDD', 'ADHD', 'SMC', 'OCD', 'TINNITUS', 'INSOMNIA', 'PARKINSON', 'DYSLEXIA',
                'ANXIETY', 'PAIN', 'CHRONIC PAIN', 'PDD NOS', 'BURNOUT', 'BIPOLAR', 'ASPERGER', 
                'DEPERSONALIZATION', 'ASD', 'WHIPLASH', 'MIGRAINE', 'EPILEPSY', 'GTS', 'PANIC', 
                'STROKE', 'TBI', 'ANOREXIA', 'CONVERSION DX', 'DPS', 'DYSPRAXIA', 'LYME', 'MSA-C', 
                'PTSD', 'TRAUMA', 'TUMOR', 'DYSCALCULIA']


    batch = 20


    test = np.loadtxt(os.path.join('/data/zhanglab/ggreiner/MENTAL/TDBRAIN', 'complete_samples_EC.csv'), delimiter=",", dtype=float)

    main_dataset = SplitDataset('complete_samples_EC.csv', '/data/zhanglab/ggreiner/MENTAL/TDBRAIN')

    res = data.random_split(main_dataset, [760,200, 2])

    train_loader = data.DataLoader(res[0], batch_size=batch, shuffle=True)
    test_loader  = data.DataLoader(res[1], batch_size=batch, shuffle=True)

    my_mental = MENTAL(60, 30, 1, batch)

    optimizer = torch.optim.Adam(my_mental.parameters(), lr)

    strs = []

    epochs = 1000

    for epoch in range(epochs):

        for (h_entry, n_entry, p_entry, label) in train_loader:

            #print(p_entry)

            h=(h_entry[0],h_entry[1])

            h[0].unsqueeze_(-1)
            h0 = h[0].transpose(1,2)
            h0 = h0.transpose(0,1)

            h[1].unsqueeze_(-1)
            h1 = h[1].transpose(1,2)
            h1 = h1.transpose(0,1)
            h1 = h1.squeeze(-1)

            h = (h0,h1)
            #print(label.size())
            #label = np.reshape(label, (20,1,1))
            count = 0
            for p in p_entry:
                count +=1
                print(p.size())
                output, h = my_mental.forward(p, n_entry, h)
            print(count)
            
            loss = torch.nn.BCELoss()
            res = loss(output, label)

            optimizer.zero_grad()
            res.backward()
            optimizer.step()
        
        if((epoch!=0) and epoch%10==0):
            correct = 0
            for (h_entry, n_entry, p_entry, label) in test_loader:

                h=(h_entry[0],h_entry[1])

                h[0].unsqueeze_(-1)
                h0 = h[0].transpose(1,2)
                h0 = h0.transpose(0,1)

                h[1].unsqueeze_(-1)
                h1 = h[1].transpose(1,2)
                h1 = h1.transpose(0,1)
                h1 = h1.squeeze(-1)

                h = (h0,h1)

                #label = np.reshape(label, (20,1,1))

                for p in p_entry:
                    output, h = my_mental.forward(p, n_entry, h)

                out = output.squeeze_(1)
                preds = []
                for i in range(0, 20):
                    for j in range(0, len(out[i])):
                        if(out[i][j] >= 0.5):
                            preds.append(1)
                        else:
                            preds.append(0)

                label = label.squeeze_(1)
                conds = []
                for i in range(0, 20):
                    for j in range(0, len(label[i])):
                        conds.append(label[i][j])

                for i in range(0, len(conds)):
                    lb = conds[i]
                    pd = preds[i]
                    if(lb==pd): 
                        correct += 1

            total = (test_loader.__len__())*batch
            s1 = "Correct : " + str(correct)
            s2 = "Total   : " + str(total)
            s3 = "Accuracy: " + str(correct/total)
            s4 = "-------------------------------------"
            s5 = "Epoch" + str(epoch)
            strs.append(s4)
            strs.append(s5)
            strs.append(s1)
            strs.append(s2)
            strs.append(s3)
            strs.append(s4)
            

    correct = 0
    for (h_entry, n_entry, p_entry, label) in test_loader:
    
        h=(h_entry[0],h_entry[1])
        
        h[0].unsqueeze_(-1)
        h0 = h[0].transpose(1,2)
        h0 = h0.transpose(0,1)

        h[1].unsqueeze_(-1)
        h1 = h[1].transpose(1,2)
        h1 = h1.transpose(0,1)
        h1 = h1.squeeze(-1)

        h = (h0,h1)

        #label = np.reshape(label, (20,1,1))

        for p in p_entry:
            output, h = my_mental.forward(p, n_entry, h)

        out = output.squeeze_(1)
        preds = []
        for i in range(0, 20):
            for j in range(0, len(out[i])):
                if(out[i][j] >= 0.5):
                    preds.append(1)
                else:
                    preds.append(0)

        label = label.squeeze_(1)
        conds = []
        for i in range(0, 20):
            for j in range(0, len(label[i])):
                conds.append(label[i][j])

        for i in range(0, len(conds)):
            lb = conds[i]
            pd = preds[i]
            if(lb==pd): 
                correct += 1

    total = (test_loader.__len__())*batch

    str4 = "________________________________"
    str1 = "Correct : " + str(correct)
    str2 = "Total   : " + str(total)
    str3 = "Accuracy: " + str(correct/total)

    strs.append(str4)
    strs.append(str1)
    strs.append(str2)
    strs.append(str3)

    with open(outfile, 'w') as f:
        for line in strs:
            f.write(line)
            f.write('\n')


run_train(lr=1e-2, outfile='1e-2.txt')
run_train(lr=2e-2, outfile='2e-2.txt')
run_train(lr=3e-2, outfile='3e-2.txt')
run_train(lr=4e-2, outfile='4e-2.txt')
run_train(lr=5e-2, outfile='5e-2.txt')

run_train(lr=1e-3, outfile='1e-3.txt')
run_train(lr=2e-3, outfile='2e-3.txt')
run_train(lr=3e-3, outfile='3e-3.txt')
run_train(lr=4e-3, outfile='4e-3.txt')
run_train(lr=5e-3, outfile='5e-3.txt')

run_train(lr=1e-4, outfile='1e-4.txt')
run_train(lr=2e-4, outfile='2e-4.txt')
run_train(lr=3e-4, outfile='3e-4.txt')
run_train(lr=4e-4, outfile='4e-4.txt')
run_train(lr=5e-4, outfile='5e-4.txt')

run_train(lr=1e-5, outfile='1e-5.txt')
run_train(lr=2e-5, outfile='2e-5.txt')
run_train(lr=3e-5, outfile='3e-5.txt')
run_train(lr=4e-5, outfile='4e-5.txt')
run_train(lr=5e-5, outfile='5e-5.txt')
