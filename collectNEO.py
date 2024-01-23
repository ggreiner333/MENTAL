import os
from os.path import dirname
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

import mne


# path of TDBRAIN
participants_path = 'TDBRAIN'

# out path
out_path = 'TDBRAIN'

def generate_NEO_Samples(ptc, out):
    survey = np.loadtxt(os.path.join(ptc, "cleaned_participants.csv"), delimiter=",", dtype=str)

    complete_samples = []
    seen = []

    for ind in survey[1:]:
        
        # Only consider individuals that we have survey data for
        # This means excluding the individuals marked for replication
        id = ind[0]
        if((not id[0] == '1') and (not seen.__contains__(id))):
            seen.append(id)

            neo = ind[6:]
            neo = [int(n) for n in neo]
            neo = np.array(neo)
            
            indication = [int(ind[2])]
            if((indication[0] != 0) and (neo[6] != -1)):
                indication = np.array(indication)
                res = np.concatenate((indication, neo))
                print(res)
                print(res.shape)
                complete_samples.append(res)
            
            
    all_complete_samples = np.array(complete_samples)
    np.save(os.path.join(out,'only_NEO_samples'), all_complete_samples)

    print("   Total samples: " + str(survey.shape[0]))
    print("Complete samples: " + str(all_complete_samples.shape[0]))()

generate_NEO_Samples(participants_path, out_path)
