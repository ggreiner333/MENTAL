import os
from os.path import dirname
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import mne
from scipy import stats


##################################################################################################
##################################################################################################
##################################################################################################

diagnoses = ['-1', 'HEALTHY', 'MDD', 'ADHD', 'SMC', 'OCD', 'TINNITUS', 'INSOMNIA', 'PARKINSON', 'DYSLEXIA',
             'ANXIETY', 'PAIN', 'CHRONIC PAIN', 'PDD NOS', 'BURNOUT', 'BIPOLAR', 'ASPERGER', 
             'DEPERSONALIZATION', 'ASD', 'WHIPLASH', 'MIGRAINE', 'EPILEPSY', 'GTS', 'PANIC', 
             'STROKE', 'TBI', 'ANOREXIA', 'CONVERSION DX', 'DPS', 'DYSPRAXIA', 'LYME', 'MSA-C', 
             'PTSD', 'TRAUMA', 'TUMOR', 'DYSCALCULIA']


# Paths for data
psd_path = 'TDBRAIN/PSD'
ptc_path = 'TDBRAIN'
out_path = 'TDBRAIN'

def normalize_data(path="/data/zhanglab/ggreiner/MENTAL/TDBRAIN"):

    # Load Data
    inds = np.load(os.path.join(path, 'small_complete_samples_EC_depression.npy'))

    normalized = np.zeros_like(inds)

    for i in range(0, inds.shape[0]):
        for j in range(0, 5):
            normalized[i][j] = inds[i][j]

    for i in range(5,inds[0].size):
        test = inds[:, i]
        z_scored = stats.zscore(test, axis=None)
        normalized[:, i] = z_scored

    for i in range(0, 4):
        print()
        print()
        for j in range(0, 100):
            print(normalized[i][j])


normalize_data()
    
