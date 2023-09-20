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

diagnoses = ['-1', 'HEALTHY', 'MDD', 'ADHD', 'SMC', 'OCD', 'TINNITUS', 'INSOMNIA', 'PARKINSON', 'DYSLEXIA',
             'ANXIETY', 'PAIN', 'CHRONIC PAIN', 'PDD NOS', 'BURNOUT', 'BIPOLAR', 'ASPERGER', 
             'DEPERSONALIZATION', 'ASD', 'WHIPLASH', 'MIGRAINE', 'EPILEPSY', 'GTS', 'PANIC', 
             'STROKE', 'TBI', 'ANOREXIA', 'CONVERSION DX', 'DPS', 'DYSPRAXIA', 'LYME', 'MSA-C', 
             'PTSD', 'TRAUMA', 'TUMOR', 'DYSCALCULIA']


batch = 20


test = np.loadtxt(os.path.join('TDBRAIN', 'complete_samples_EC.csv'), delimiter=",", dtype=float)

main_dataset = SplitDataset('complete_samples_EC.csv', 'TDBRAIN')

#print(main_dataset.__len__())

res = data.random_split(main_dataset, [760,200, 2])

train_loader = data.DataLoader(res[0], batch_size=batch, shuffle=True)
test_loader  = data.DataLoader(res[1], batch_size=batch)

my_mental = MENTAL(60, 30, 36, batch)

optimizer = torch.optim.Adam(my_mental.parameters(), lr=1e-6, weight_decay=1e-9)

#print("parameters : ")
#print(list(my_mental.parameters()))

epochs = 1000

for epoch in range(epochs):

    for (d_entry, n_entry, p_entry, label) in train_loader:

        h = (d_entry, d_entry)

        h[0].unsqueeze_(-1)
        h0 = h[0].transpose(1,2)
        h0 = h0.transpose(0,1)

        h[1].unsqueeze_(-1)
        h1 = h[1].transpose(1,2)
        h1 = h1.transpose(0,1)
        h1 = h1.squeeze(-1)

        h = (h0,h1)

        label = np.reshape(label, (20,1,36))
        #print(label.shape)
        #print(label)

        for p in p_entry:
            output, h = my_mental.forward(p, n_entry, h)

        #print(output.shape)

        #output = output.squeeze_(1)
        #print(output)
        #print(output.size())

        #print(label)
        #print(label.size())
        
        loss = torch.nn.MSELoss()
        res = loss(output, label)

        optimizer.zero_grad()
        res.backward()
        optimizer.step()

    print("-----------------------")
    print("Epoch: " + str(epoch))
    print(" Loss: " + str(res) )
    print("-----------------------")


    if((epoch!=0) and epoch%10==0):
        correct = 0
        for (d_entry, n_entry, p_entry, label) in test_loader:
            h = (d_entry, d_entry)

            h[0].unsqueeze_(-1)
            h0 = h[0].transpose(1,2)
            h0 = h0.transpose(0,1)

            h[1].unsqueeze_(-1)
            h1 = h[1].transpose(1,2)
            h1 = h1.transpose(0,1)
            h1 = h1.squeeze(-1)

            h = (h0,h1)

            label = np.reshape(label, (20,1,36))
            #print(label.shape)
            #print(label)

            for p in p_entry:
                output, h = my_mental.forward(p, n_entry, h)

            #print(output)
            preds = []
            for i in range(0, 20):
                maxIdx = 0
                #print(output[i])
                sum = 0
                for j in range(0, len(output[i][1])):
                    if output[i][1][j] >= output[i][1][maxIdx]:
                        maxIdx=j
                    sum = sum + output[i][1][j]
                preds.append(j)
                maxIdx=0
                print(sum)


            conds = []
            for i in range(0, 20):
                for j in range(0, len(label[i][1])):
                    if(int(label[i][1][j]) > 0):
                        conds.append(int(label[i][1][j]))
                        break
            
            for i in range(0, len(conds)):
                lb = conds[i]
                pd = preds[i]
                print("----------------------------------------------")
                print("Condition : " + str(lb))
                print("Prediction: " + str(pd))
                print("Equal?    : " + str(lb==pd))
                if(lb==pd): 
                    correct += 1

        total = (test_loader.__len__())*batch

        print("------------------------------------")
        print("------------------------------------")
        print("------------------------------------")
        print("------------------------------------")

        print("Correct : " + str(correct))
        print("Total   : " + str(total))
        print("Accuracy: " + str(correct/total))
    



correct = 0
for (d_entry, n_entry, p_entry, label) in test_loader:
   
    h = (d_entry, d_entry)

    h[0].unsqueeze_(-1)
    h0 = h[0].transpose(1,2)
    h0 = h0.transpose(0,1)

    h[1].unsqueeze_(-1)
    h1 = h[1].transpose(1,2)
    h1 = h1.transpose(0,1)
    h1 = h1.squeeze(-1)

    h = (h0,h1)

    label = np.reshape(label, (20,1,36))

    for p in p_entry:
        output, h = my_mental.forward(p, n_entry, h)

    #print(output)
    preds = []
    for i in range(0, 20):
        maxIdx = 0
        #print(output[i])
        sum = 0
        for j in range(0, len(output[i][1])):
            if output[i][1][j] >= output[i][1][maxIdx]:
                maxIdx=j
            sum = sum + output[i][1][j]
        preds.append(j)
        maxIdx=0
        print(sum)


    conds = []
    for i in range(0, 20):
        for j in range(0, len(label[i][1])):
            if(int(label[i][1][j]) > 0):
                conds.append(int(label[i][1][j]))
                break
    
    for i in range(0, len(conds)):
        lb = conds[i]
        pd = preds[i]
        print("----------------------------------------------")
        print("Condition : " + str(lb))
        print("Prediction: " + str(pd))
        print("Equal?    : " + str(lb==pd))
        if(lb==pd): 
            correct += 1

total = (test_loader.__len__())*batch
print("------------------------------------")
print("------------------------------------")
print("------------------------------------")
print("------------------------------------")
print("Correct : " + str(correct))
print("Total   : " + str(total))
print("Accuracy: " + str(correct/total))