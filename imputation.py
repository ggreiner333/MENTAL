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

print(test[0])

# Create Dataset and Dataset Loader
mm_dataset = MultiModalDataset('complete_samples_EC.csv', 'TDBRAIN')
dataset_loader = data.DataLoader(mm_dataset, batch_size=128, shuffle=True)

# Create an instance of the encoder
my_encoder = VAE(7864, 128)

optimizer = torch.optim.Adam(my_encoder.parameters(), lr=1e-2, weight_decay=1e-9)


epochs = 20


for epoch in range(epochs):

    for (entry, label) in dataset_loader:

        output = my_encoder.forward(entry)

        loss = torch.nn.MSELoss()
        res = loss(output, entry)

        optimizer.zero_grad()
        res.backward()
        optimizer.step()

        print("-----------------------")
        print("Epoch: " + epoch)
        print(" Loss: " + loss )
        print("-----------------------")

