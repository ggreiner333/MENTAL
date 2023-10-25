import os
from os.path import dirname
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
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
psd_path = 'TDBRAIN/PSD_all'
#ptc_path = 'data/zhanglab/ggreiner/MENTAL/TDBRAIN/'
ptc_path = 'TDBRAIN'
#out_path = 'data/zhanglab/ggreiner/MENTAL/TDBRAIN/samples'
out_path = 'TDBRAIN'


def count_missing(path="C:\\Users\\glgre\\Documents\\ResearchCode\\MENTAL\\TDBRAIN"):

    # Load Demographic and Survey Data

    inds = np.loadtxt(os.path.join(path, "cleaned_participants.csv"), delimiter=",", dtype=str)

    # bins: [0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90]

    bins = []
    for i in range(0,len(diagnoses)):
        vals = []
        for i in range(0,65):
            vals.append(0)
        bins.append(vals)

    labels = ["", "count"]
    for i in inds[0][3:]:
        labels.append(i)

    for b in range(0, len(bins)):
        bins[b][0] = diagnoses[b]

    for row in inds[1:]:
        bin = int(row[2])
        for i in range(3,len(row)):
            if(row[i] != "-1"):
                #print(row[i])
                bins[bin][i-1] = bins[bin][i-1]+1
        bins[bin][1] = bins[bin][1]+1

    out = []

    out.append(labels)
    for count in bins:
        out.append(count)

    final = np.asarray(out)
    np.savetxt(os.path.join(path,'data_count.csv'), final, delimiter=',', fmt="%s")


def plot_missing(path="C:\\Users\\glgre\\Documents\\ResearchCode\\MENTAL\\TDBRAIN"):

    # Load Demographic and Survey Data

    sns.set_theme()

    counts = pd.read_csv(os.path.join(path, "short_data_count_label.csv"))
    #print(counts)

    column_headers = list(counts.columns)[:-1]
    #print(column_headers)

    index = pd.Index(['NEO-FFI', 'Age', 'Education', 'Gender', 'Samples'])
    counts_test = counts.set_index(index)
    counts_test = counts_test.transpose()
    #print(counts_test)


    f, ax = plt.subplots(2, 1, figsize=(10, 22), gridspec_kw={'height_ratios': [1, 1.0/37]})
    #f, ax = plt.subplots(2, 1, figsize=(10, 20))

    myMask = counts_test['Samples'] == 1301
    #print(counts_test[myMask])

    myMask2 = counts_test['Samples'] != 1301
    #print(counts_test[myMask2])

    x_axis_labels = ['NEO-FFI', 'Age', 'Education', 'Gender', 'Samples']

    d = sns.heatmap(counts_test[myMask2], annot=True, fmt="d", linewidth=0.5, square=False, yticklabels=column_headers, xticklabels=False, ax=ax[0], cmap="Reds", cbar=False, vmin=0, vmax=400)
    d.set_yticklabels(d.get_yticklabels(), rotation=0, fontsize=13)
    t = sns.heatmap(counts_test[myMask] ,  annot=True, fmt="d", linewidth=0.5, square=False, yticklabels=['Total'], xticklabels=x_axis_labels, ax=ax[1], cmap="Greens", cbar=False, vmin=0, vmax=1301)
    t.set_yticklabels(t.get_yticklabels(), rotation=0, fontsize=13)
    t.set_xticklabels(t.get_xticklabels(), rotation=0, fontsize=13)

    #ax[1].set_xlabel('Data of Interest', fontsize = 12)
    ax[0].set_title('Presence of Recorded Data Among Disorders', fontsize = 16)
    ax[0].set_ylabel('Disorder', fontsize=16)

    plt.tight_layout()
    plt.subplots_adjust(hspace=0)
    plt.show()


#count_missing()
plot_missing()

def count_gender_disorder(gen, path="C:\\Users\\glgre\\Documents\\ResearchCode\\MENTAL\\TDBRAIN"):

    # Load Demographic and Survey Data

    inds = np.loadtxt(os.path.join(path, "cleaned_participants.csv"), delimiter=",", dtype=str)

    # bins: [0,10,20,30,40,50,60,70,80,90]

    # 0 - female
    # 1 - male

    labels = ["0-10","10-20","20-30","30-40","40-50","50-60","60-70","70-80","80-90","90+", "Missing"]

    gen_bins = []
    for i in range(0,len(diagnoses)+1):
        vals = []
        for i in range(0,len(labels)):
            vals.append(0)
        gen_bins.append(vals)

    last = len(diagnoses)

    for row in inds[1:]:
        if(row[4] == gen):
            bin = int(row[2])
            if(row[3] != "-1"):
                age = float(row[3])
                idx = int(age/10)
                bin = int(row[2])
                gen_bins[bin][idx] = gen_bins[bin][idx]+1
                gen_bins[last][idx] = gen_bins[last][idx]+1
            else:
                gen_bins[bin][10] = gen_bins[bin][10]+1
                gen_bins[last][10] = gen_bins[last][10]+1
        

    out = []

    out.append(labels)
    for count in gen_bins:
        out.append(count)

    final = np.asarray(out)
    np.savetxt(os.path.join(path,'gender'+gen+'_disorder_count.csv'), final, delimiter=',', fmt="%s")

#count_gender_disorder("0")
#count_gender_disorder("1")

def plot_male_disorder(path="C:\\Users\\glgre\\Documents\\ResearchCode\\MENTAL\\TDBRAIN"):

    # Load Demographic and Survey Data

    sns.set_theme()

    ######################################
    #        Male
    ######################################


    male = pd.read_csv(os.path.join(path, "gender1_disorder_count.csv"))

    Mcol_heads = list(male.columns)[:-1]

    index = pd.Index(['MDD', 'ADHD', 'SMC', 'OCD', 'Tinnitus', 'Insomnia', 'Parkinsons', 'Dyslexia',
             'Anxiety', 'Pain', 'Chronic Pain', 'PDD NOS', 'Burnout', 'Bipolar', 'Asperger', 
             'Depersonalization', 'ASD', 'Whiplash', 'Migraine', 'Epilepsy', 'GTS', 'Panic', 
             'Stroke', 'TBI', 'Anorexia', 'Conversion DX', 'DPS', 'Dyspraxia', 'LYME', 'MSA-C', 
             'PTSD', 'Trauma', 'Tumor', 'Dyscalculia', 'Healthy', 'Missing Label',  'Total'])
    
    male_idx = male.set_index(index)
    #print(male_idx)

    x_tick_labels = ["0-10","10-20","20-30","30-40","40-50","50-60","60-70","70-80","80-90", "Missing Age"]

    myMask = male_idx['Missing Age'] == 6
    #print(male_idx[myMask])

    myMask2 = male_idx['Missing Age'] != 6
    #print(male_idx[myMask2])

    f, ax = plt.subplots(2, 1, figsize=(11, 22), gridspec_kw={'height_ratios': [1, 1.0/37]})


    d = sns.heatmap(male_idx[myMask2], annot=True, fmt="d", linewidth=0.5, square=False, xticklabels=False, ax=ax[0], cmap="Reds", cbar=False, vmin=0, vmax=50)
    d.set_yticklabels(d.get_yticklabels(), rotation=0, fontsize=13)
    
    t = sns.heatmap(male_idx[myMask] ,  annot=True, fmt="d", linewidth=0.5, square=False, xticklabels=x_tick_labels, ax=ax[1], cmap="Greens", cbar=False, vmin=0, vmax=118)
    t.set_yticklabels(t.get_yticklabels(), rotation=0, fontsize=13)
    t.set_xticklabels(t.get_xticklabels(), rotation=0, fontsize=13)

    ax[1].set_xlabel('Age Range (Years)', fontsize = 16)
    ax[0].set_title('Disorder Distribution Among Male Samples', fontsize = 16)
    ax[0].set_ylabel('Disorder', fontsize=16)

    #plt.tight_layout()
    plt.subplots_adjust(hspace=0)
    plt.show()

def plot_female_disorder(path="C:\\Users\\glgre\\Documents\\ResearchCode\\MENTAL\\TDBRAIN"):

    # Load Demographic and Survey Data

    sns.set_theme()

    ######################################
    #        Female
    ######################################


    female = pd.read_csv(os.path.join(path, "gender0_disorder_count.csv"))

    Fcol_heads = list(female.columns)[:-1]

    index = pd.Index(['MDD', 'ADHD', 'SMC', 'OCD', 'Tinnitus', 'Insomnia', 'Parkinsons', 'Dyslexia',
             'Anxiety', 'Pain', 'Chronic Pain', 'PDD NOS', 'Burnout', 'Bipolar', 'Asperger', 
             'Depersonalization', 'ASD', 'Whiplash', 'Migraine', 'Epilepsy', 'GTS', 'Panic', 
             'Stroke', 'TBI', 'Anorexia', 'Conversion DX', 'DPS', 'Dyspraxia', 'LYME', 'MSA-C', 
             'PTSD', 'Trauma', 'Tumor', 'Dyscalculia', 'Healthy', 'Missing Label', 'Total'])

    female_idx = female.set_index(index)
    #print(female_idx)

    x_tick_labels = ["0-10","10-20","20-30","30-40","40-50","50-60","60-70","70-80","80-90", "Missing Age"]

    myMask = female_idx['Missing Age'] == 15
    #print(female_idx[myMask])

    myMask2 = female_idx['Missing Age'] != 15
    #print(female_idx[myMask2])

    f, ax = plt.subplots(2, 1, figsize=(11, 22), gridspec_kw={'height_ratios': [1, 1.0/37]})


    d = sns.heatmap(female_idx[myMask2], annot=True, fmt="d", linewidth=0.5, square=False, xticklabels=False, ax=ax[0], cmap="Oranges", cbar=False, vmin=0, vmax=57)
    d.set_yticklabels(d.get_yticklabels(), rotation=0, fontsize=13)
    t = sns.heatmap(female_idx[myMask] ,  annot=True, fmt="d", linewidth=0.5, square=False, xticklabels=x_tick_labels, ax=ax[1], cmap="Purples", cbar=False, vmin=0, vmax=118)
    t.set_yticklabels(t.get_yticklabels(), rotation=0, fontsize=13)
    t.set_xticklabels(t.get_xticklabels(), rotation=0, fontsize=13)

    ax[1].set_xlabel('Age Range (Years)', fontsize = 16)
    ax[0].set_title('Disorder Distribution Among Female Samples', fontsize = 16)
    ax[0].set_ylabel('Disorder', fontsize=16)

    '''
    d.axvline(x =  0, color = 'k', linewidth = 3)
    d.axvline(x = 10, color = 'k', linewidth = 3) 
    d.axhline(y =  0, color = 'k', linewidth = 3)

    t.axvline(x =  0, color = 'k', linewidth = 3)
    t.axvline(x = 10, color = 'k', linewidth = 3) 
    t.axhline(y =  1, color = 'k', linewidth = 3)
    '''

    #plt.tight_layout()
    plt.subplots_adjust(hspace=0)
    plt.show()

plot_male_disorder()
plot_female_disorder()