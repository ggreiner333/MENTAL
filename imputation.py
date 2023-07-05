import os
from os.path import dirname
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

import mne


##################################################################################################
##################################################################################################
##################################################################################################



# Paths for data

psd_path = 'data/zhanglab/ggreiner/MENTAL/TDBRAIN/PSD'
ptc_path = 'data/zhanglab/ggreienr/MENTAL/TDBRAIN/derivatives'


# Load Demographic and Survey Data

inds = np.loadtxt(os.path.join(ptc_path, "participants.csv"), delimiter=",", dtype=str)

# Grab the individual's ids

useful_ids = []

for i in inds:
    useful_ids.append(i(0))




