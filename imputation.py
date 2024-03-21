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
data_loader = data.DataLoader(complete_dataset, batch_size=20, shuffle=True)

# Create an instance of the encoder
encoder = VAE(INPUT_DIM, Z_DIM)

optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-3)

epochs = 2

for epoch in range(epochs):

    for vals in data_loader:

        output, mu, var = encoder.forward(vals)

        recon_loss = torch.nn.MSELoss()
        loss = recon_loss(output, vals)

        kl_loss = - torch.sum(1 + torch.log(var.pow(2))-mu.pow(2)-var.pow(2))

        loss = loss + kl_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print("-----------------------")
    print("Epoch: " + str(epoch))
    print(" Loss: " + str(loss) )
    print("-----------------------")


individuals = np.load(os.path.join('TDBRAIN', 'small_missing_samples_EC_adhd.npy'))
imputed = []

for ind in individuals:
    if(ind[1] != (-1.0)):
        mask = np.ones(ind.size, dtype='double')
        missing = np.zeros_like(ind, dtype='double')
        missing[0] = 1
        for i in range(1, ind.size):
            if(ind[i]==(-1)):
                mask[i] = 0
                missing[i] = 1
        
        masked = ind*mask
        masked = torch.from_numpy(masked)
        out = encoder.forward(masked[1:])
        imputed_ind = missing*(out.numpy())
        
        imputed.append(ind+imputed_ind)
    else:
        print(ind[0:10])

imputed = np.array(imputed)

np.save(os.path.join('TDBRAIN','small_imputed_samples_EC_adhd.npy'), imputed)

    


