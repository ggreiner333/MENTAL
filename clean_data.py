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
out_path = 'TDBRAIN'

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
    survey = np.loadtxt(os.path.join(ptc, "cleaned_participants.csv"), delimiter=",", dtype=str)
    
    samples_EO = []
    samples_EC = []

    samples_EO.append(np.asarray(survey[0]))
    samples_EC.append(np.asarray(survey[0]))

    for ind in survey[1:]:
        
        # Only consider individuals that we have survey data for
        # This means excluding the individuals marked for replication
        id = ind[0]
        if(not id[0] == '1'):

            # Navigate to the directory with the psd information
            loc = os.path.join(psd, id)
            files = os.listdir(loc)

            for f in files:

                # Load the PSD values from the files
                pth = os.path.join(loc,f)
                psds = np.load(pth, allow_pickle=True)
                psds = np.squeeze(psds)
                psds = psds.flatten()

                # Combine survey and PSD data
                combined = np.concatenate(ind,psds)

                print(combined)

                if(f.__contains__("EC")):
                    samples_EC.append(np.asarray(combined))
                else:
                    samples_EO.append(np.asarray(combined))

                print(psds.shape)
                print(psds)

    # Save the combined samples into csv files

    all_combined_EC = np.array(samples_EC)
    np.save(os.path.join(out,'combined_samples_EC.csv'), all_combined_EC, allow_pickle=True)

    all_combined_EO = np.array(samples_EO)
    np.save(os.path.join(out,'combined_samples_EO.csv'), all_combined_EO, allow_pickle=True)


def generate_split():
    ind_path = 'TDBRAIN/preprocessed'
    all = os.listdir(ind_path)
    print(len(all))


#generate_samples(ptc_path, psd_path, out_path)

generate_split()       


