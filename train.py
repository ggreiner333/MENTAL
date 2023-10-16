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

def run_train(learn_rate, wd, outfile):
    diagnoses = ['-1', 'HEALTHY', 'MDD', 'ADHD', 'SMC', 'OCD', 'TINNITUS', 'INSOMNIA', 'PARKINSON', 'DYSLEXIA',
                'ANXIETY', 'PAIN', 'CHRONIC PAIN', 'PDD NOS', 'BURNOUT', 'BIPOLAR', 'ASPERGER', 
                'DEPERSONALIZATION', 'ASD', 'WHIPLASH', 'MIGRAINE', 'EPILEPSY', 'GTS', 'PANIC', 
                'STROKE', 'TBI', 'ANOREXIA', 'CONVERSION DX', 'DPS', 'DYSPRAXIA', 'LYME', 'MSA-C', 
                'PTSD', 'TRAUMA', 'TUMOR', 'DYSCALCULIA']


    batch_sz = 10


    #test = np.loadtxt(os.path.join('/data/zhanglab/ggreiner/MENTAL/TDBRAIN', 'small_complete_samples_EC_depression.npy'), delimiter=",", dtype=float)

    main_dataset = SplitDataset('small_complete_samples_EC_depression.npy', '/data/zhanglab/ggreiner/MENTAL/TDBRAIN')

    res = data.random_split(main_dataset, [560, 140, 5])

    train_loader = data.DataLoader(res[0], batch_size=batch_sz, shuffle=True)
    test_loader  = data.DataLoader(res[1], batch_size=batch_sz, shuffle=True)

    my_mental = MENTAL(60, 30, 1, batch_sz)


    optimizer = torch.optim.Adam(my_mental.parameters(), lr=learn_rate, weight_decay=wd)

    strs = []

    epochs = 1000

    for epoch in range(epochs):

        for (h_entry, n_entry, p_entry, label) in train_loader:

            #h=(h_entry[0],h_entry[1])

            #h[0].unsqueeze_(-1)
            #h0 = h[0].transpose(1,2)
            #h0 = h0.transpose(0,1)

            #h[1].unsqueeze_(-1)
            #h1 = h[1].transpose(1,2)
            #h1 = h1.transpose(0,1)
            #h1 = h1.squeeze(-1)

            h = h_entry.transpose(0,1)
            #print(label.size())
            #print(label)
            label_reshaped = np.reshape(label, (batch_sz,1,1))
            #print(label_reshaped)
            
            #batches = np.array()
            count = 0

            test = []

            for i in range(0, 60):
                batch = []
                for j in range(0,10):
                    cur = p_entry[j][i]
                    arr_cur = np.asarray(cur)
                    batch.append(arr_cur)
                test.append(batch)

            formatted = np.array(test)
            
            psd_tensor = torch.from_numpy(formatted)

            #print(psd_tensor.shape)

            for p in psd_tensor:
                output, h_res = my_mental.forward(p, n_entry, h)
                h = h_res
            
            #print(output)

            loss = torch.nn.BCELoss()
            res = loss(output, label_reshaped)

            optimizer.zero_grad()
            res.backward()
            optimizer.step()
        
        if((epoch!=0) and epoch%1==0):
            correct = 0
            for (h_entry, n_entry, p_entry, label) in test_loader:

                #h=(h_entry[0],h_entry[1])

                #h[0].unsqueeze_(-1)
                #h0 = h[0].transpose(1,2)
                #h0 = h0.transpose(0,1)

                #h[1].unsqueeze_(-1)
                #h1 = h[1].transpose(1,2)
                #h1 = h1.transpose(0,1)
                #h1 = h1.squeeze(-1)

                h = h_entry.transpose(0,1)
                #print(label.size())
                #print(label)
                label_reshaped = np.reshape(label, (batch_sz,1,1))
                #print(label_reshaped)
                
                #batches = np.array()
                count = 0

                test = []

                for i in range(0, 60):
                    batch = []
                    for j in range(0,10):
                        cur = p_entry[j][i]
                        arr_cur = np.asarray(cur)
                        batch.append(arr_cur)
                    test.append(batch)

                formatted = np.array(test)
                
                psd_tensor = torch.from_numpy(formatted)

                #print(psd_tensor.shape)

                for p in psd_tensor:
                    output, h_res = my_mental.forward(p, n_entry, h)
                    h = h_res

                out = output.squeeze_(1)
                #print(out)
                preds = []
                for i in range(0, batch_sz):
                    for j in range(0, len(out[i])):
                        if(out[i][j] >= 0.5):
                            preds.append(1)
                        else:
                            preds.append(0)

                label = label.squeeze_(1)
                print(label)
                conds = []
                for i in range(0, batch_sz):
                    for j in range(0, len(label[i])):
                        conds.append(label[i][j])

                for i in range(0, len(conds)):
                    lb = conds[i]
                    pd = preds[i]
                    if(lb==pd): 
                        correct += 1

            total = (test_loader.__len__())*batch_sz
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
            print(s4)
            print(s1)
            print(s2)
            print(s3)
            print(s4)
            

    correct = 0
    for (h_entry, n_entry, p_entry, label) in test_loader:
    
        #h=(h_entry[0],h_entry[1])

        #h[0].unsqueeze_(-1)
        #h0 = h[0].transpose(1,2)
        #h0 = h0.transpose(0,1)

        #h[1].unsqueeze_(-1)
        #h1 = h[1].transpose(1,2)
        #h1 = h1.transpose(0,1)
        #h1 = h1.squeeze(-1)

        h = h_entry.transpose(0,1)
        #print(label.size())
        #print(label)
        label_reshaped = np.reshape(label, (batch_sz,1,1))
        #print(label_reshaped)
        
        #batches = np.array()
        count = 0

        test = []

        for i in range(0, 60):
            batch = []
            for j in range(0,10):
                cur = p_entry[j][i]
                arr_cur = np.asarray(cur)
                batch.append(arr_cur)
            test.append(batch)

        formatted = np.array(test)
        
        psd_tensor = torch.from_numpy(formatted)

        #print(psd_tensor.shape)

        for p in psd_tensor:
            output, h_res = my_mental.forward(p, n_entry, h)
            h = h_res

        out = output.squeeze_(1)
        preds = []
        for i in range(0, batch_sz):
            for j in range(0, len(out[i])):
                if(out[i][j] >= 0.5):
                    preds.append(1)
                else:
                    preds.append(0)

        label = label.squeeze_(1)
        conds = []
        for i in range(0, batch_sz):
            for j in range(0, len(label)):
                conds.append(label[i][j])

        for i in range(0, len(conds)):
            lb = conds[i]
            pd = preds[i]
            if(lb==pd): 
                correct += 1

    total = (test_loader.__len__())*batch_sz

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


run_train(learn_rate=1e-4, wd=1e-6, outfile='we-6_le-4_small.txt')
run_train(learn_rate=1e-4, wd=1e-7, outfile='we-7_le-4_small.txt')
run_train(learn_rate=1e-4, wd=1e-8, outfile='we-8_le-4_small.txt')

run_train(learn_rate=1e-5, wd=1e-6, outfile='we-6_le-5_small.txt')
run_train(learn_rate=1e-5, wd=1e-7, outfile='we-7_le-5_small.txt')
run_train(learn_rate=1e-5, wd=1e-8, outfile='we-8_le-5_small.txt')

run_train(learn_rate=1e-6, wd=1e-6, outfile='we-6_le-6_small.txt')
run_train(learn_rate=1e-6, wd=1e-7, outfile='we-7_le-6_small.txt')
run_train(learn_rate=1e-6, wd=1e-8, outfile='we-8_le-6_small.txt')
