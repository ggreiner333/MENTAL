import os
from os.path import dirname
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

import mne


##################################################################################################
##################################################################################################
##################################################################################################

# path of preprocessed EEG data
preprocess_file_path = 'TDBRAIN/preprocessed'

# path of directory where we will save the PSD features
psds_path = 'TDBRAIN/PSD_all'

# channels
all_included = [ "Fp1",  "Fp2",   "F7",   "F3",    "Fz",    "F4",   "F8", "FC3", "FCz", "FC4", "T7", "C3", "Cz",
                  "C4",   "T8",  "CP3",  "CPz",   "CP4",    "P7",   "P3",  "Pz",  "P4",  "P8", "O1", "Oz", "O2" ]

##################################################################################################
##################################################################################################
##################################################################################################


def get_ind():
    ec_psds = np.load('/data/zhanglab/ggreiner/MENTAL/TDBRAIN/small_complete_samples_EC_depression.npy', allow_pickle=True)

    for ind in ec_psds[0:3]:
        psds = ind[65:]
        res = []
        for i in range(0,26):
            res.append(psds[i*5])
        res = np.array(res)

        plt.plot(res)
        plt.show()
        

get_ind()
