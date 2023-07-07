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

    complete_samples = []
    missing_samples = []

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
                combined = np.concatenate((ind,psds))

                if(f.__contains__("EC")):
                    samples_EC.append(np.asarray(combined))
                else:
                    samples_EO.append(np.asarray(combined))


    # Save the combined samples into csv files

    all_combined_EC = np.array(samples_EC, dtype=object)
    np.save(os.path.join(out,'combined_samples_EC'), all_combined_EC, allow_pickle=True)

    all_combined_EO = np.array(samples_EO, dtype=object)
    np.save(os.path.join(out,'combined_samples_EO'), all_combined_EO, allow_pickle=True)



def separate_missing_samples(ptc, psd, out):
    survey = np.loadtxt(os.path.join(ptc, "cleaned_participants.csv"), delimiter=",", dtype=str)
    
    missing_samples = []
    complete_samples = []

    for ind in survey[1:]:
        
        # Only consider individuals that we have survey data for
        # This means excluding the individuals marked for replication
        id = ind[0]
        if(not id[0] == '1'):

            # Navigate to the directory with the psd information
            loc = os.path.join(psd, id)
            files = os.listdir(loc)

            found = False
            for f in files:
                if(f.__contains__("EC")):
                    found = True
                    # Load the PSD values from the files
                    pth = os.path.join(loc,f)
                    psds = np.load(pth, allow_pickle=True)
                    psds = np.squeeze(psds)
                    psds = psds.flatten()

                    # Combine survey and PSD data
                    combined = np.asarray(np.concatenate((ind,psds)))
                    if(combined.__contains__('-1')):
                        missing_samples.append(combined)
                    else:
                        complete_samples.append(combined)

            if(not found):
                combo = np.asarray(np.concatenate((ind, np.zeros(7800))))
                missing_samples.append(combo)
            
    all_complete_samples = np.array(complete_samples, dtype=object)
    np.savetxt(os.path.join(out,'complete_samples_EC.csv'), all_complete_samples, delimiter=',', fmt="%s")

    all_missing_samples = np.array(missing_samples, dtype=object)
    np.savetxt(os.path.join(out,'missing_samples_EC.csv'), all_missing_samples, delimiter=',', fmt="%s")

    print("   Total samples: " + str(survey.shape[0]))
    print("Complete samples: " + str(all_complete_samples.shape[0]))
    print(" Missing samples: " + str(all_missing_samples.shape[0]))


def load_attempt(path):

    loadeds = np.load(path, allow_pickle=True)

    loaded = loadeds[1]

    print("ID: " + loaded[0])
    print("sess ID: " + loaded[1])

    print("Diagnosis: " + loaded[2])

    print("Age, gender, education: ", loaded[3:6])

    print("NEO-FFI: " + loaded[6:30])

    print(loaded)

separate_missing_samples(ptc_path, psd_path, out_path)
#generate_samples(ptc_path, psd_path, out_path)


load_attempt('TDBRAIN/complete_samples_EC.npy')




