import os
from os.path import dirname
from pathlib import Path

import numpy as np
import torch
from torch.utils import data

import mne


from Model.dataset import MultiModalDataset
from Model.dataset import ImputingDataset
from Model.dataset import ImputingMissingDataset
from Model.vae import VAE
from Model.vae import VAE_Both

##################################################################################################
##################################################################################################
##################################################################################################

INPUT_DIM = 7864
Z_DIM = 512


# Create Dataset and Dataset Loader
complete_dataset = ImputingDataset('small_complete_samples_EC_top5.npy', '/data/zhanglab/ggreiner/MENTAL/TDBRAIN')
data_loader = data.DataLoader(complete_dataset, batch_size=5, shuffle=True)

# Create an instance of the encoder
encoder = VAE(INPUT_DIM, Z_DIM)

optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-3)

epochs = 50

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


missing_dataset = ImputingMissingDataset('small_missing_samples_EC_top5.npy', '/data/zhanglab/ggreiner/MENTAL/TDBRAIN')
missing_data_loader = data.DataLoader(missing_dataset, batch_size=1, shuffle=True)
imputed = []

for (ind, mask, missing) in missing_data_loader:
    masked = ind*mask
    masked = masked.type(torch.float32)
    test = masked.size()

    out = encoder.forward(masked[0][1:])
    imputed_ind = torch.mul(missing[0][1:], out[0])
    
    filled = masked[0][1:]+imputed_ind
    filled = filled.detach().numpy()
    test = [ind[0][0].detach().numpy()]
    test = np.array(test)
    filled = np.array(filled)
    res = np.concatenate([test, filled])

    imputed.append(res)

imputed = np.array(imputed)
print(imputed.shape)

np.save(os.path.join('/data/zhanglab/ggreiner/MENTAL/TDBRAIN','small_imputed_samples_EC_top5.npy'), imputed)