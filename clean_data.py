import os
from os.path import dirname
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
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



def clean_individuals(path="C:\\Users\\glgre\\Documents\\ResearchCode\\MENTAL\\TDBRAIN"):

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

def clean_individuals_depression_or_not(path="/data/zhanglab/ggreiner/MENTAL/TDBRAIN"):

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
    
        found = False
        missing = False
        for d in disorders:
            if(diagnoses.index(d.strip()) == 2):
                res = []
                for info in i:
                    res.append(info)
                res[2] = diagnoses.index(d.strip())
                samples.append(res)
                found = True
            if(diagnoses.index(d.strip()) == 0):
                missing = True
        if((not found) and (not missing)):
            res = []
            for info in i:
                res.append(info)
            res[2] = 0
            samples.append(res)
        
    final = np.asarray(samples)
    #print(final)
    np.savetxt(os.path.join(path,'cleaned_participants_depression.csv'), final, delimiter=',', fmt="%s")

def clean_individuals_ADHD(path="/data/zhanglab/ggreiner/MENTAL/TDBRAIN"):

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
    
        found = False
        missing = False
        for d in disorders:
            if(diagnoses.index(d.strip()) == 3):
                res = []
                for info in i:
                    res.append(info)
                res[2] = diagnoses.index(d.strip())
                samples.append(res)
                found = True
            if(diagnoses.index(d.strip()) == 0):
                missing = True
        if((not found) and (not missing)):
            res = []
            for info in i:
                res.append(info)
            res[2] = 0
            samples.append(res)
        
    final = np.asarray(samples)
    #print(final)
    np.savetxt(os.path.join(path,'cleaned_participants_adhd.csv'), final, delimiter=',', fmt="%s")


def clean_individuals_Health_Non(path="/data/zhanglab/ggreiner/MENTAL/TDBRAIN"):

    # Load Demographic and Survey Data

    inds = np.loadtxt(os.path.join(path, "participants.csv"), delimiter=",", dtype=str)

    samples = []

    cols = []
    for name in inds[0]:
        cols.append(name)
    samples.append(cols)

    count = 0

    for i in inds[1:]:
        id = i[0]
        disorders = (i[2].upper()).split("/")
    
        found = False
        missing = False
        for d in disorders:
            diag = diagnoses.index(d.strip())
            if(diag == 1 or (diag == 3 and count < 92)):
                res = []
                for info in i:
                    res.append(info)
                res[2] = diag
                samples.append(res)
                found = True
                if(diag == 3):
                    count+=1
        
    final = np.asarray(samples)
    #print(final)
    np.savetxt(os.path.join(path,'cleaned_participants_health_adhd.csv'), final, delimiter=',', fmt="%s")

#clean_individuals_Health_Non()

def generate_samples(ptc, psd, out):
    survey = np.loadtxt(os.path.join(ptc, "cleaned_participants_depression.csv"), delimiter=",", dtype=str)
    
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
        #print(id)
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
    #print(all_combined_EC)
    #np.save(os.path.join(out,'combined_samples_EC'), all_combined_EC, allow_pickle=True)

    all_combined_EO = np.array(samples_EO, dtype=object)
    #np.save(os.path.join(out,'combined_samples_EO'), all_combined_EO, allow_pickle=True)

#generate_samples(ptc_path, psd_path, out_path)

def separate_missing_samples_EC(ptc, psd, out):
    survey = np.loadtxt(os.path.join(ptc, "cleaned_participants_health_adhd.csv"), delimiter=",", dtype=str)
    
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

            sn = ind[1]
            found = False
            for f in files:
                if(f.__contains__("EC") and f.__contains__(sn)):
                    found = True
                    # Load the PSD values from the files
                    pth = os.path.join(loc,f)
                    psds = np.load(pth, allow_pickle=True)
                    psds = np.squeeze(psds)
                    psds = psds.flatten()

                    # Combine survey and PSD data
                    combined = np.asarray(np.concatenate((ind[1:],psds)), dtype="float32")
                    combined[0] = float((ind[0].split("-"))[1])+(int(sn)/10)
                    if(combined.__contains__(-1)):
                        missing_samples.append(combined)
                    else:
                        complete_samples.append(combined)

            if(not found):
                combo = np.concatenate((ind[1:], np.zeros(7800)))
                combo[0] = float((ind[0].split("-"))[1])+(int(sn)/10)
                combo = np.asarray(combo, dtype="float32")
                missing_samples.append(combo)
            
    all_complete_samples = np.array(complete_samples)
    np.save(os.path.join(out,'small_complete_samples_EC_health_adhd'), all_complete_samples)

    print(complete_samples)
    all_missing_samples = np.array(missing_samples)
    np.save(os.path.join(out,'small_missing_samples_EC_health_adhd'), all_missing_samples)

    print("   Total samples: " + str(survey.shape[0]))
    print("Complete samples: " + str(all_complete_samples.shape[0]))
    print(" Missing samples: " + str(all_missing_samples.shape[0]))


def separate_missing_samples_EO(ptc, psd, out):
    survey = np.loadtxt(os.path.join(ptc, "cleaned_participants_health_adhd.csv"), delimiter=",", dtype=str)
    
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

            sn = ind[1]
            found = False
            for f in files:
                if(f.__contains__("EO") and f.__contains__(sn)):
                    found = True
                    # Load the PSD values from the files
                    pth = os.path.join(loc,f)
                    psds = np.load(pth, allow_pickle=True)
                    psds = np.squeeze(psds)
                    psds = psds.flatten()

                    # Combine survey and PSD data
                    combined = np.asarray(np.concatenate((ind[1:],psds)), dtype="float32")
                    combined[0] = float((ind[0].split("-"))[1])+(int(sn)/10)
                    if(combined.__contains__(-1)):
                        missing_samples.append(combined)
                    else:
                        complete_samples.append(combined)

            if(not found):
                combo = np.concatenate((ind[1:], np.zeros(7800)))
                combo[0] = float((ind[0].split("-"))[1])+(int(sn)/10)
                combo = np.asarray(combo, dtype="float32")
                missing_samples.append(combo)
            
    all_complete_samples = np.array(complete_samples)
    np.save(os.path.join(out,'small_complete_samples_EO_health_adhd'), all_complete_samples)

    print(complete_samples)
    all_missing_samples = np.array(missing_samples)
    np.save(os.path.join(out,'small_missing_samples_EO_health_adhd'), all_missing_samples)

    print("   Total samples: " + str(survey.shape[0]))
    print("Complete samples: " + str(all_complete_samples.shape[0]))
    print(" Missing samples: " + str(all_missing_samples.shape[0]))


def separate_missing_samples_EO_EC(ptc, psd, out):
    survey = np.loadtxt(os.path.join(ptc, "cleaned_participants_health_adhd.csv"), delimiter=",", dtype=str)
    
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

            sn = ind[1]
            found_ec = False
            found_eo = False
            for f in files:
                if(f.__contains__("EO") and f.__contains__(sn)):
                    found_eo = True
                    # Load the PSD values from the files
                    pth = os.path.join(loc,f)
                    eo_psds = np.load(pth, allow_pickle=True)
                    eo_psds = np.squeeze(eo_psds)
                    eo_psds = eo_psds.flatten()
                if(f.__contains__("EC") and f.__contains__(sn)):
                    found_ec = True
                    # Load the PSD values from the files
                    pth = os.path.join(loc,f)
                    ec_psds = np.load(pth, allow_pickle=True)
                    ec_psds = np.squeeze(ec_psds)
                    ec_psds = ec_psds.flatten()

            if(found_ec and found_eo):
                psds = np.concatenate((ec_psds, eo_psds))
                # Combine survey and PSD data
                combined = np.asarray(np.concatenate((ind[1:],psds)), dtype="float32")
                combined[0] = float((ind[0].split("-"))[1])+(int(sn)/10)
                if(combined.__contains__(-1)):
                    missing_samples.append(combined)
                else:
                    complete_samples.append(combined)

            if(not (found_ec and found_eo)):
                combo = np.concatenate((ind[1:], np.zeros(7800*2)))
                combo[0] = float((ind[0].split("-"))[1])+(int(sn)/10)
                combo = np.asarray(combo, dtype="float32")
                missing_samples.append(combo)
            
    all_complete_samples = np.array(complete_samples)
    print(all_complete_samples.shape)
    np.save(os.path.join(out,'small_complete_samples_EC_EO_health_adhd'), all_complete_samples)

    print(complete_samples)
    all_missing_samples = np.array(missing_samples)
    np.save(os.path.join(out,'small_missing_samples_EC_EO_health_adhd'), all_missing_samples)

    print("   Total samples: " + str(survey.shape[0]))
    print("Complete samples: " + str(all_complete_samples.shape[0]))
    print(" Missing samples: " + str(all_missing_samples.shape[0]))

def load_attempt(path):

    loadeds = np.loadtxt(path, delimiter=",")

    loaded = loadeds[1]

    print(loaded[0])

    print(loaded)


def find_min_max_gender():
    tsvdat = pd.read_csv("C:\\Users\\glgre\\Documents\\ResearchCode\\MENTAL\\TDBRAIN\\participants.csv")

    maleages = np.unique(tsvdat['age'][tsvdat['gender']==1])
    femaleages = np.unique(tsvdat['age'][tsvdat['gender']==0])

    minM = np.where(maleages > 0, maleages, np.inf).min()
    minF = np.where(femaleages > 0, femaleages, np.inf).min()
    min_Male = maleages.min()
    max_Male = maleages.max()

    min_Female = femaleages.min()
    max_Female = femaleages.max()

    print("Min Male: " + str(minM) + ", Max Male: " + str(max_Male))
    print("Min Female: " + str(minF) + ", Max Female: " + str(max_Female))

#separate_missing_samples_EC(ptc_path, psd_path, out_path)
#separate_missing_samples_EO(ptc_path, psd_path, out_path)
#separate_missing_samples_EO_EC(ptc_path, psd_path, out_path)
#generate_samples(ptc_path, psd_path, out_path)


#load_attempt('TDBRAIN/small_complete_samples_EC.csv')

#find_min_max_gender()


def get_disorders_for_analysis(path="/data/zhanglab/ggreiner/MENTAL/TDBRAIN"):

    # Load Demographic and Survey Data

    inds = np.loadtxt(os.path.join(path, "participants.csv"), delimiter=",", dtype=str)

    samples = []

    for i in inds[1:]:
        id = i[0]
        disorders = (i[2].upper()).split("/")
    
        found = False
        missing = False
        for d in disorders:
            if(diagnoses.index(d.strip()) == 2):
                res = [id]
                res.append(i[1])
                res.append(diagnoses.index(d.strip()))
                samples.append(res)
                found = True
            if(diagnoses.index(d.strip()) == 0):
                missing = True
        if((not found) and (not missing)):
            res = [id]
            res.append(i[1])
            res.append(diagnoses.index(disorders[0].strip()))
            samples.append(res)

    final = np.asarray(samples)
    np.savetxt(os.path.join(path,'individuals_disorders.csv'), final, delimiter=',', fmt="%s")

def create_disorder_psd_EC(ptc, psd, out):
    survey = np.loadtxt(os.path.join(ptc, "individuals_disorders.csv"), delimiter=",", dtype=str)
    
    missing_samples = []
    complete_samples = []

    for ind in survey:
        
        # Only consider individuals that we have survey data for
        # This means excluding the individuals marked for replication
        id = ind[0]
        if(not id[0] == '1'):

            # Navigate to the directory with the psd information
            loc = os.path.join(psd, id)
            files = os.listdir(loc)

            sn = ind[1]
            found = False
            for f in files:
                if(f.__contains__("EC") and f.__contains__(sn)):
                    found = True
                    # Load the PSD values from the files
                    pth = os.path.join(loc,f)
                    psds = np.load(pth, allow_pickle=True)
                    psds = np.squeeze(psds)
                    psds = psds.flatten()

                    # Combine survey and PSD data
                    combined = np.asarray(np.concatenate((ind[2:],psds)), dtype="float32")
                    complete_samples.append(combined)

            if(not found):
                combo = np.concatenate((ind[2:], np.zeros(7800)))
                combo = np.asarray(combo, dtype="float32")
                missing_samples.append(combo)
            
    all_complete_samples = np.array(complete_samples)
    np.save(os.path.join(out,'disorders_EC_psds'), all_complete_samples)

    print("   Total samples: " + str(survey.shape[0]))
    print("Complete samples: " + str(all_complete_samples.shape[0]))


#get_disorders_for_analysis()
#create_disorder_psd_EC(ptc_path, psd_path, out_path)

def create_disorder_psd_EO(ptc, psd, out):
    survey = np.loadtxt(os.path.join(ptc, "individuals_disorders.csv"), delimiter=",", dtype=str)
    
    missing_samples = []
    complete_samples = []

    for ind in survey:
        
        # Only consider individuals that we have survey data for
        # This means excluding the individuals marked for replication
        id = ind[0]
        if(not id[0] == '1'):

            # Navigate to the directory with the psd information
            loc = os.path.join(psd, id)
            files = os.listdir(loc)

            sn = ind[1]
            found = False
            for f in files:
                if(f.__contains__("EO") and f.__contains__(sn)):
                    found = True
                    # Load the PSD values from the files
                    pth = os.path.join(loc,f)
                    psds = np.load(pth, allow_pickle=True)
                    psds = np.squeeze(psds)
                    psds = psds.flatten()

                    # Combine survey and PSD data
                    combined = np.asarray(np.concatenate((ind[2:],psds)), dtype="float32")
                    complete_samples.append(combined)

            if(not found):
                combo = np.concatenate((ind[2:], np.zeros(7800)))
                combo = np.asarray(combo, dtype="float32")
                missing_samples.append(combo)
            
    all_complete_samples = np.array(complete_samples)
    np.save(os.path.join(out,'disorders_EO_psds'), all_complete_samples)

    print("   Total samples: " + str(survey.shape[0]))
    print("Complete samples: " + str(all_complete_samples.shape[0]))

#create_disorder_psd_EO(ptc_path, psd_path, out_path)
    
def combine_imputed_complete():
    complete = np.load(os.path.join('TDBRAIN','small_complete_samples_EO_depression.npy'))
    imputed = np.load(os.path.join('TDBRAIN','small_imputed_samples_EO_depression.npy'))

    combined = []
    for c in complete:
        combined.append(c)
    for i in imputed:
        combined.append(i)
    
    combined = np.array(combined)
    print(combined.size)
    print(combined)

    np.save(os.path.join('TDBRAIN','small_imputed_complete_samples_EO_depression.npy'), combined)

#combine_imputed_complete()
    
def testing():
    imputed1 = np.load(os.path.join('TDBRAIN','small_imputed_samples_EC_adhd.npy'))
    imputed2 = np.load(os.path.join('TDBRAIN','small_imputed_samples_EO_adhd.npy'))
    missing = np.load(os.path.join('TDBRAIN','small_missing_samples_EC_EO_adhd.npy'))
    print(missing.shape)
    
    new_imp = []
    for m in missing:
        first = m[0:4]
        middle = m[5:65]
        last = m[66:]
        found = False
        for ind in imputed1:
            if(ind[0] == m[0]):
                middle = ind[5:65]
                found = True
        if(not found):
            for ind in imputed2:
                if(ind[0] == m[0]):
                    middle = ind[5:65]
                    found = True
        first = np.array(first)
        middle = np.array(middle)
        last = np.array(last)
        res = np.concatenate([first, middle, last])
        if(found):
            new_imp.append(res)
    
    new_imp = np.array(new_imp)
    
    print(new_imp)
    print(new_imp.shape)

    np.save(os.path.join('TDBRAIN','small_imputed_samples_EC_EO_adhd.npy'), new_imp)

testing()
