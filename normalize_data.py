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
    inds = np.load(os.path.join(path, 'small_imputed_complete_samples_EC_EO_adhd.npy'))

    normalized = np.zeros_like(inds)

    for i in range(0, inds.shape[0]):
        for j in range(0, 5):
            normalized[i][j] = inds[i][j]

    for i in range(5,inds[0].size):
        test = inds[:, i]
        z_scored = stats.zscore(test, axis=None)
        normalized[:, i] = z_scored

    print(normalized.shape)

    np.save(file=os.path.join(path, 'normalized_small_imputed_complete_samples_EC_EO_adhd.npy'), arr=normalized)

def normalize_data_EO(path="/data/zhanglab/ggreiner/MENTAL/TDBRAIN"):

    # Load Data
    inds = np.load(os.path.join(path, 'small_complete_samples_EO_health_adhd.npy'))

    normalized = np.zeros_like(inds)

    for i in range(0, inds.shape[0]):
        for j in range(0, 5):
            normalized[i][j] = inds[i][j]

    for i in range(5,inds[0].size):
        test = inds[:, i]
        z_scored = stats.zscore(test, axis=None)
        normalized[:, i] = z_scored

    print(normalized.shape)

    np.save(file=os.path.join(path, 'normalized_small_complete_samples_EO_health_adhd.npy'), arr=normalized)

def normalize_data_EC_EO(path="/data/zhanglab/ggreiner/MENTAL/TDBRAIN"):

    # Load Data
    inds = np.load(os.path.join(path, 'small_complete_samples_EC_EO_health_adhd.npy'))

    normalized = np.zeros_like(inds)

    for i in range(0, inds.shape[0]):
        for j in range(0, 5):
            normalized[i][j] = inds[i][j]

    for i in range(5,inds[0].size):
        test = inds[:, i]
        z_scored = stats.zscore(test, axis=None)
        normalized[:, i] = z_scored

    print(normalized.shape)

    np.save(file=os.path.join(path, 'normalized_small_complete_samples_EC_EO_health_adhd.npy'), arr=normalized)


def test_load(path="/data/zhanglab/ggreiner/MENTAL/TDBRAIN"):
    inds = np.load(os.path.join(path, 'only_EC_adhd_healthy_samples.npy'))
    hcount = 0
    dcount = 0
    for i in inds:
        v = int(i[0])
        if(v == 1):
            hcount+=1
        else:
            dcount+=1
            
    print(f"Healthy individuals: {hcount}")
    print(f"Disorder individuals: {dcount}")
    print(inds.shape)

#test_load()

normalize_data()
#normalize_data_EO()
#normalize_data_EC_EO()