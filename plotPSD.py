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


##################################################################################################
##################################################################################################
##################################################################################################


def get_ind():
    ec_psds = np.load('/data/zhanglab/ggreiner/MENTAL/TDBRAIN/small_complete_samples_EC_depression.npy', allow_pickle=True)

    for ind in ec_psds[0:20]:
        print(ind[0:70])

get_ind()
