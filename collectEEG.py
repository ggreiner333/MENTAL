import os
from os.path import dirname
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

import mne

# path of preprocessed EEG data
preprocess_file_path = 'TDBRAIN/preprocessed'

# path of directory where we will save the PSD features
psds_path = 'TDBRAIN/PSD'

# path of TDBRAIN
participants_path = 'TDBRAIN'

# out path
out_path = 'TDBRAIN'

def generate_EC_Samples(ptc, psd, out):
    survey = np.loadtxt(os.path.join(ptc, "cleaned_participants.csv"), delimiter=",", dtype=str)

    complete_samples = []
    seen = []

    for ind in survey[1:]:
        
        # Only consider individuals that we have survey data for
        # This means excluding the individuals marked for replication
        id = ind[0]
        if((not id[0] == '1') and (not seen.__contains__(id))):
            seen.append(id)
            # Navigate to the directory with the psd information
            loc = os.path.join(psd, id)
            files = os.listdir(loc)

            sn = ind[1]
            for f in files:
                if(f.__contains__("EC") and f.__contains__(sn)):
                    found = True
                    # Load the PSD values from the files
                    pth = os.path.join(loc,f)
                    psds = np.load(pth, allow_pickle=True)
                    psds = np.squeeze(psds)
                    psds = psds.flatten()

            indication = [int(ind[2])]
            if(indication[0] != 0):
                indication = np.array(indication)
                res = np.concatenate((indication, psds))
                complete_samples.append(res)
            
            
    all_complete_samples = np.array(complete_samples)
    np.save(os.path.join(out,'only_EC_samples'), all_complete_samples)

    print("   Total samples: " + str(survey.shape[0]))
    print("Complete samples: " + str(all_complete_samples.shape[0]))

def generate_EO_Samples(ptc, psd, out):
    survey = np.loadtxt(os.path.join(ptc, "cleaned_participants.csv"), delimiter=",", dtype=str)

    complete_samples = []
    seen = []

    for ind in survey[1:]:
        
        # Only consider individuals that we have survey data for
        # This means excluding the individuals marked for replication
        id = ind[0]
        if((not id[0] == '1') and (not seen.__contains__(id))):
            seen.append(id)

            # Navigate to the directory with the psd information
            loc = os.path.join(psd, id)
            files = os.listdir(loc)

            sn = ind[1]
            for f in files:
                if(f.__contains__("EO") and f.__contains__(sn)):
                    found = True
                    # Load the PSD values from the files
                    pth = os.path.join(loc,f)
                    psds = np.load(pth, allow_pickle=True)
                    psds = np.squeeze(psds)
                    psds = psds.flatten()

            indication = [int(ind[2])]
            if(indication[0] != 0):
                indication = np.array(indication)
                res = np.concatenate((indication, psds))
                complete_samples.append(res)
            
            
    all_complete_samples = np.array(complete_samples)
    np.save(os.path.join(out,'only_EO_samples'), all_complete_samples)

    print("   Total samples: " + str(survey.shape[0]))
    print("Complete samples: " + str(all_complete_samples.shape[0]))

def generate_EC_EO_Samples(ptc, psd, out):
    survey = np.loadtxt(os.path.join(ptc, "cleaned_participants.csv"), delimiter=",", dtype=str)

    complete_samples = []
    seen = []
    for ind in survey[1:]:
        
        # Only consider individuals that we have survey data for
        # This means excluding the individuals marked for replication
        id = ind[0]
        if((not id[0] == '1') and (not seen.__contains__(id))):
            seen.append(id)
            # Navigate to the directory with the psd information
            loc = os.path.join(psd, id)
            files = os.listdir(loc)

            sn = ind[1]
            found_eo = False
            found_ec = False
            for f in files:
                if(f.__contains__("EO") and f.__contains__(sn)):
                    found_eo = True
                    # Load the PSD values from the files
                    pth = os.path.join(loc,f)
                    eo_psds = np.load(pth, allow_pickle=True)
                    eo_psds = np.squeeze(eo_psds)
                    eo_psds = eo_psds.flatten()
                elif(f.__contains__("EC") and f.__contains__(sn)):
                    found_ec = True
                    # Load the PSD values from the files
                    pth = os.path.join(loc,f)
                    ec_psds = np.load(pth, allow_pickle=True)
                    ec_psds = np.squeeze(ec_psds)
                    ec_psds = ec_psds.flatten()

            indication = [int(ind[2])]
            if((indication[0] != 0)):
                if(found_eo and found_ec):
                        indication = np.array(indication)
                        psds = np.concatenate((ec_psds, eo_psds))
                        res = np.concatenate((indication, psds))
                        complete_samples.append(res)
            
            
    all_complete_samples = np.array(complete_samples)
    np.save(os.path.join(out,'only_EC_EO_samples'), all_complete_samples)

    print("   Total samples: " + str(survey.shape[0]))
    print("Complete samples: " + str(all_complete_samples.shape[0]))

generate_EC_Samples(participants_path, psds_path, out_path) 
generate_EO_Samples(participants_path, psds_path, out_path) 
generate_EC_EO_Samples(participants_path, psds_path, out_path) 