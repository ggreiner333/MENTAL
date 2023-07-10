import os
from os.path import dirname
from pathlib import Path

import numpy as np
import torch
from torch.utils import data

import mne


from Model.dataset import MultiModalDataset
from Model.mentalModel import MENTAL


##################################################################################################
##################################################################################################
##################################################################################################

test = np.loadtxt(os.path.join('TDBRAIN', 'complete_samples_EC.csv'), delimiter=",", dtype=float)

# Create Dataset and Dataset Loader
mm_dataset = MultiModalDataset('complete_samples_EC.csv', 'TDBRAIN')
dataset_loader = data.DataLoader(mm_dataset, batch_size=20, shuffle=True)

my_mental = MENTAL(60, 30, 15)

optimizer = torch.optim.Adam(my_mental.parameters(), lr=1e-7, weight_decay=1e-9)

epochs = 1000

for epoch in range(epochs):

    for (entry, label) in dataset_loader:

        output = my_mental.forward(entry)

        loss = torch.nn.MSELoss()
        res = loss(output, entry)

        optimizer.zero_grad()
        res.backward()
        optimizer.step()

    print("-----------------------")
    print("Epoch: " + str(epoch))
    print(" Loss: " + str(res) )
    print("-----------------------")

