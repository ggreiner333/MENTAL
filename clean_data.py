import os
from os.path import dirname
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

import mne


##################################################################################################
##################################################################################################
##################################################################################################

diagnoses = ['-1', 'HEALTHY', 'MDD', 'ADHD', 'SMC', 'OCD', 'TINNITUS', 'INSOMNIA', 'PARKINSON', 'DYSLEXIA',
             'ANXIETY', 'PAIN', 'CHRONIC PAIN', 'PDD NOS', 'BURNOUT', 'BIPOLAR', 'ASPERGER', 
             'DEPERSONALIZATION', 'ASD', 'WHIPLASH', 'MIGRAINE', 'EPILEPSY', 'GTS', 'PANIC', 
             'STROKE', 'TBI', 'ANOREXIA', 'CONVERSION DX', 'DPS', 'DYSPRAXIA', 'LYME', 'MSA-C', 
             'PTSD', 'TRAUMA', 'TUMOR', 'DYSCALCULIA']


# Paths for data

#psd_path = 'data/zhanglab/ggreiner/MENTAL/TDBRAIN/PSD'
psd_path = 'TDBRAIN/PSD'
#ptc_path = 'data/zhanglab/ggreiner/MENTAL/TDBRAIN/'
ptc_path = 'TDBRAIN'
#out_path = 'data/zhanglab/ggreiner/MENTAL/TDBRAIN/samples'
out_path = 'TDBRAIN/samples'

def clean_individuals(path):

    # Load Demographic and Survey Data

    inds = np.loadtxt(os.path.join(path, "participants.csv"), delimiter=",", dtype=str)

    samples = []

    cols = []
    for name in inds[0]:
        cols.append(name)
    samples.append(cols)

    for i in inds[1:]:
        id = i[0]
        disorders = (i[2].upper()).split("/")
        
        for d in disorders:
            res = []
            for info in i:
                res.append(info)
            res[2] = diagnoses.index(d.strip())
            samples.append(res)
        
    final = np.asarray(samples)
    np.savetxt(os.path.join(path,'cleaned_participants.csv'), final, delimiter=',', fmt="%s")

def generate_samples(ptc, psd, out):
    #survey = np.loadtxt(os.path.join(ptc, "cleaned_participants.csv"), delimiter=",", dtype=str)
    
    #data = survey[1]
    #id = data[0]
    id = 'sub-19681349'
    
    loc = os.path.join(psd, id)
    files = os.listdir(loc)
    for f in files:
        pth = os.path.join(loc,f)
        test = np.load(pth, allow_pickle=True)
        print(test)

generate_samples(ptc_path, psd_path, out_path)
        


