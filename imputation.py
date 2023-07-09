import os
from os.path import dirname
from pathlib import Path

import numpy as np
import torch
from torch.utils import data

import mne


from Model.dataset import MultiModalDataset
from Model.vae import VAE

##################################################################################################
##################################################################################################
##################################################################################################

test = np.loadtxt(os.path.join('TDBRAIN', 'complete_samples_EC.csv'), delimiter=",", dtype=float)

# Create Dataset and Dataset Loader
mm_dataset = MultiModalDataset('complete_samples_EC.csv', 'TDBRAIN')
dataset_loader = data.DataLoader(mm_dataset, batch_size=20, shuffle=True)

# Create Missing Dataset
mis_dataset = MultiModalDataset('missing_samples_EC.csv', 'TDBRAIN')
mis_dataset_loader = data.DataLoader(mis_dataset, batch_size=20, shuffle=True)

# Create an instance of the encoder
my_encoder = VAE(7864, 128)

optimizer = torch.optim.Adam(my_encoder.parameters(), lr=1e-6, weight_decay=1e-9)

epochs = 30


for epoch in range(epochs):

    for (entry, label) in dataset_loader:

        output = my_encoder.forward(entry)

        loss = torch.nn.MSELoss()
        res = loss(output, entry)

        optimizer.zero_grad()
        res.backward()
        optimizer.step()

    print("-----------------------")
    print("Epoch: " + str(epoch))
    print(" Loss: " + str(res) )
    print("-----------------------")

predictions = []

for (entry, label) in mis_dataset_loader:
    res = my_encoder(entry)
    predictions.append(res)

print(predictions)
