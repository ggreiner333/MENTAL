import os
from os.path import dirname
from pathlib import Path

import numpy as np
import torch
from torch.utils import data

import mne


from Model.dataset import MultiModalDataset
from Model.dataset import ImputingDataset
from Model.vae import VAE

##################################################################################################
##################################################################################################
##################################################################################################

INPUT_DIM = 7864
Z_DIM = 256


# Create Dataset and Dataset Loader
complete_dataset = ImputingDataset('small_complete_samples_EC_adhd.npy', 'TDBRAIN')
data_loader = data.DataLoader(complete_dataset, batch_size=5, shuffle=True)

# Create an instance of the encoder
encoder = VAE(INPUT_DIM, Z_DIM)

optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-7)

epochs = 1000

for epoch in range(epochs):

    for vals in data_loader:

        output, mu, var = encoder.forward(vals)

        recon_loss = torch.nn.BCELoss()
        loss = recon_loss(output, vals)

        print(output)
        print(vals)

        kl_loss = - torch.sum(1 + torch.log(var.pow(2))-mu.pow(2)-var.pow(2))

        loss = loss + kl_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print("-----------------------")
    print("Epoch: " + str(epoch))
    print(" Loss: " + str(loss) )
    print("-----------------------")


