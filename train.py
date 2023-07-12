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

print(main_dataset.__len__())

res = data.random_split(main_dataset, [760,200, 2])

train_loader = data.DataLoader(res[0], batch_size=batch, shuffle=True)
test_loader  = data.DataLoader(res[1], batch_size=batch)

my_mental = MENTAL(60, 3, 3, batch)

optimizer = torch.optim.Adam(my_mental.parameters(), lr=1e-6, weight_decay=1e-9)

epochs = 1

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

        for p in p_entry:
            output, h = my_mental.forward(p, n_entry, h)

        output = output.squeeze_(1)
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

for (d_entry, n_entry, p_entry, label) in test_loader:
    out, h = my_mental(p, n_entry, h)
    #print(out)
    preds = []
    for i in range(0, 20):
        idx = 0
        min = abs(out[i][0][0]-1.0)
        for j in range(1, 35):
            dist = abs(out[i][0][j]-1.0)
            if(dist < min):
                idx = j
                min = dist
        temp = torch.zeros([35])
        temp[int(idx)] = 1.0
        preds.append(temp)
        


    for i in range(0, label.size()[0]):
        print("----------------------------------------------")
        print("Condition : " + str(label[i]))
        print("Prediction: " + str(preds[i]))
