import os
from os.path import dirname
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import mne

from sklearn.metrics import roc_curve, auc


##################################################################################################
##################################################################################################
##################################################################################################

diagnoses = ['-1', 'HEALTHY', 'MDD', 'ADHD', 'SMC', 'OCD', 'TINNITUS', 'INSOMNIA', 'PARKINSON', 'DYSLEXIA',
             'ANXIETY', 'PAIN', 'CHRONIC PAIN', 'PDD NOS', 'BURNOUT', 'BIPOLAR', 'ASPERGER', 
             'DEPERSONALIZATION', 'ASD', 'WHIPLASH', 'MIGRAINE', 'EPILEPSY', 'GTS', 'PANIC', 
             'STROKE', 'TBI', 'ANOREXIA', 'CONVERSION DX', 'DPS', 'DYSPRAXIA', 'LYME', 'MSA-C', 
             'PTSD', 'TRAUMA', 'TUMOR', 'DYSCALCULIA']

# path of preprocessed EEG data
preprocess_file_path = 'TDBRAIN/preprocessed'

# path of directory where we will save the PSD features
psds_path = 'TDBRAIN/PSD_all'

# channels
all_included = [ "Fp1",  "Fp2",   "F7",   "F3",    "Fz",    "F4",   "F8", "FC3", "FCz", "FC4", "T7", "C3", "Cz",
                  "C4",   "T8",  "CP3",  "CPz",   "CP4",    "P7",   "P3",  "Pz",  "P4",  "P8", "O1", "Oz", "O2" ]

frontal   = np.arange(0,10, 1)
temporal  = [10,14]
central   = [11,12,13,15,16,17]
parietal  = [18,19,20,21,22]
occupital = [23,24,25]

##################################################################################################
##################################################################################################
##################################################################################################

def test():
    ec_psds = np.load('small_complete_samples_EC_depression.npy', allow_pickle=True)
    print(ec_psds.shape)
    seen = []
    other = []
    for ind in ec_psds:
        cur = ind[0]
        if(seen.__contains__(cur)):
            print("duped")
            print(ind[0:7])
            print(other[seen.index(cur)])
        else:
            seen.append(cur)
            other.append(ind[0:5])

def test2():
    ec_psds = np.load('small_complete_samples_EC_depression.npy', allow_pickle=True)
    seen = []
    other = []
    for ind in ec_psds:
        cur = ind[2]
        print(cur)
        if(seen.__contains__(cur)):
            print("duped")
            print(ind[0:7])
            print(other[seen.index(cur)])
        else:
            seen.append(cur)
            other.append(ind[0:5])

def plot_box():
    ec_psds = np.load('small_complete_samples_EC_depression.npy', allow_pickle=True)

    d_count = 0
    depressed = []
    for i in range(0, 26):
        depressed.append([])

    o_count = 0
    other = []
    for i in range(0, 26):
        other.append([])

    for ind in ec_psds:
        psds = ind[65:]
        total = np.zeros(26)
        for i in range(0,60):
            res = []
            for j in range(0,26):
                res.append(psds[(i*130)+j*5+2])
            res = np.array(res)
            total = total+res
        total = total/60

        if(ind[1] == 0.0):
            o_count += 1
            for i in range(0, 26):
                other[i].append(total[i])
        else:
            d_count += 1
            for i in range(0, 26):
                depressed[i].append(total[i])
    

    #, showfliers=False
    x_pos_range = np.arange(2)
    x_pos = (x_pos_range * 0.5) + 0.75

    bp1=plt.boxplot(
        depressed, sym='', widths=0.3, labels=all_included, patch_artist=True,notch=True,
        positions=[x_pos[0] + j*1 for j in range(0, 26)]
    )
    bp2=plt.boxplot(
        other, sym='', widths=0.3,patch_artist=True, notch=True,
        positions=[x_pos[1] + j*1 for j in range(0, 26)]
    )

    for box in bp1['boxes']:
        box.set_facecolor("red")
    
    plt.xticks(ticks=np.arange(1,27,1), labels=all_included)
    plt.legend([bp1["boxes"][0], bp2["boxes"][0]], ['Depressed', 'Other'], loc='upper right')
    plt.show()

def get_ind():
    ec_psds = np.load('small_complete_samples_EC_depression.npy', allow_pickle=True)

    total_ct = 0
    total_nd = np.zeros(26)

    total_dct = 0
    total_d = np.zeros(26)

    for ind in ec_psds:
        psds = ind[65:]
        total = np.zeros(26)
        for i in range(0,60):
            res = []
            for j in range(0,26):
                res.append(psds[(i*130)+j*5+1])
            res = np.array(res)
            total = total+res
        total = total/60

        if(ind[1] == 0.0):
            total_ct += 1
            total_nd = total_nd + total
        else:
            total_dct +=1
            total_d = total_d + total

    total_nd = total_nd/total_ct
    total_d = total_d/total_dct
    plt.plot(total_nd, label="not depressed")
    plt.plot(total_d, label="depressed")
    plt.xticks(ticks=np.arange(0,26,1), labels=all_included)
    plt.legend(loc='upper right')
    plt.show()

def plot_test():
    accs = np.load('diff_MENTAL_EC_EO_ACCS_epoch_400.npy', allow_pickle=True)

    labels = np.arange(0, 401, 1)
    plt.figure(figsize=(15,10))
    plt.plot(labels, accs)
    plt.title("Accuracy of diff EC+EO MENTAL for 400 epochs, batch size 15")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.yticks(ticks=np.arange(0,1.01,0.1))

    plt.savefig("diff_mental_epoch400_b15_w6_l3_accuracy_ec_eo")
    plt.clf()


##################################################################################################
##################################################################################################
##################################################################################################

bands = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']


def mdd_healthy_means_multiple(band, num_disorders, closed, outliers):
    if(closed):
        all_psds = np.load('disorders_EC_psds.npy', allow_pickle=True)
    else:
        all_psds = np.load('disorders_EO_psds.npy', allow_pickle=True)

    index = bands.index(band)

    all = []

    counts = [0 for i in range(0, num_disorders)]

    for i in range(0,num_disorders):
        all.append(np.zeros(26))

    ct = 0
    for ind in all_psds:
        disorder = int(ind[0])-1
        psds = ind[1:]
        total = np.zeros(26)
        for i in range(0,60):
            res = []
            for j in range(0,26):
                val = psds[(i*130)+j*5+index]
                if(outliers):
                    res.append(val)
                elif(disorder==0 and j==14 and index>=3):
                    res.append(val if(val < 20) else 0)
                else:
                    res.append(val if(val <= 1000) else 0)
            res = np.array(res)
            total = total+res
        total = total/60

        if(disorder < num_disorders and disorder >= 0):
            counts[disorder] += 1
            all[disorder] = all[disorder] + total

    mx = 0
    mn = 200
    for i in range(0, num_disorders):
        all[i] = all[i]/counts[i]
        if(np.max(all[i]) > mx):
            mx = np.max(all[i])
        if(np.min(all[i]) < mn):
            mn = np.min(all[i])
    
    for i in range(0, num_disorders):
        vals = []
        for item in all[i]:
            vals.append((item-mn)/(mx-mn))
        vals = np.array(vals)
        plt.plot(vals, label=diagnoses[i+1])
        plt.scatter(np.arange(0,26,1), vals, s=10)

    plt.title(band+" PSD " + ("EC" if closed else "EO"), fontsize=18)
    plt.xticks(ticks=np.arange(0,26,1), labels=all_included)
    plt.legend(bbox_to_anchor=(1.0, 1.0), loc='upper left')
    plt.xlabel("Channel", fontsize=14)
    plt.ylabel("PSD", fontsize=14)
    plt.tight_layout()
    plt.show()

#for b in bands:
#    mdd_healthy_means_multiple(b, 3, closed=True, outliers=False)



def mdd_healthy_means(band, d1, d2, closed, outliers):
    if(closed):
        all_psds = np.load('disorders_EC_psds.npy', allow_pickle=True)
    else:
        all_psds = np.load('disorders_EO_psds.npy', allow_pickle=True)

    index = bands.index(band)
    d1_idx = diagnoses.index(d1)
    d2_idx = diagnoses.index(d2)

    total_ct = 0
    total_nd = np.zeros(26)

    total_dct = 0
    total_d = np.zeros(26)

    for ind in all_psds:
        disorder = int(ind[0])-1
        psds = ind[1:]
        total = np.zeros(26)
        for i in range(0,60):
            res = []
            for j in range(0,26):
                val = psds[(i*130)+j*5+index]
                if(outliers):
                    res.append(val)
                elif(disorder==0 and (j==10 or j==14) and (index==4)):
                    res.append(val if(val < 20) else 0)
                else:
                    res.append(val if(val <= 1000) else 0)
            res = np.array(res)
            total = total+res
        total = total/60

        if(ind[0] == d1_idx):
            total_ct += 1
            total_nd = total_nd + total
        elif(ind[0] == d2_idx):
            total_dct +=1
            total_d = total_d + total

    total_nd = total_nd/total_ct
    total_d = total_d/total_dct

    dc = 'orange'
    nc = 'green'

    plt.plot(total_nd, label=d1, c=nc)
    plt.plot(total_d, label=d2, c=dc)
    plt.scatter(np.arange(0,26,1), total_nd, s=10, color=nc)
    plt.scatter(np.arange(0,26,1), total_d, s=10, color=dc)
    plt.xticks(ticks=np.arange(0,26,1), labels=all_included)
    plt.title(band+" PSD " + ("EC" if closed else "EO"), fontsize=18)
    plt.legend(bbox_to_anchor=(1.0, 1.0), loc='upper left')
    plt.xlabel("Channel", fontsize=16)
    plt.ylabel("PSD", fontsize=16)
    plt.tight_layout()
    plt.show()

#for b in bands:
#    mdd_healthy_means(b, 'HEALTHY', 'MDD', closed=False, outliers=False)



def mdd_healthy_ranges(band, d1, d2, closed):
    if(closed):
        all_psds = np.load('disorders_EC_psds.npy', allow_pickle=True)
    else:
        all_psds = np.load('disorders_EO_psds.npy', allow_pickle=True)

    index = bands.index(band)
    d1_idx = diagnoses.index(d1)
    d2_idx = diagnoses.index(d2)

    d_count = 0
    depressed = []
    for i in range(0, 26):
        depressed.append([])

    o_count = 0
    other = []
    for i in range(0, 26):
        other.append([])

    for ind in all_psds:
        psds = ind[1:]
        total = []
        for i in range(0, 26):
            total.append([])
        for i in range(0,60):
            for j in range(0,26):
                total[j].append(psds[(i*130)+j*5+index])

        if(ind[0] == d1_idx):
            o_count += 1
            for i in range(0, 26):
                for j in total[i]:
                    other[i].append(j)
        elif(ind[0] == d2_idx):
            d_count += 1
            for i in range(0, 26):
                for j in total[i]:
                    depressed[i].append(j)
    

    dMeans = []
    dStds = []
    dStds.append([])
    dStds.append([])
    dPairs = []
    for x in depressed:
        dMeans.append(np.median(x))
        res  = np.percentile(x, [75 ,25])
        dStds[0].append(res[0])
        dStds[1].append(res[1])
        dPairs.append(res)

    oMeans = []
    oStds = []
    oStds.append([])
    oStds.append([])
    oPairs = []
    for x in other:
        oMeans.append(np.median(x))
        res  = np.percentile(x, [75 ,25])
        oStds[0].append(res[0])
        oStds[1].append(res[1])
        oPairs.append(res)

    dStds = np.array(dStds)
    print(dStds)

    oStds = np.array(oStds)

    dc = 'blue'
    nc = 'red'

    fig, ax = plt.subplots()
    plt.scatter(np.arange(0,26,1), dStds[0], marker='_', c='b', s=40)
    plt.scatter(np.arange(0,26,1), dStds[1], marker='_', c='b', s=40)
    plt.scatter(np.arange(0,26,1), oStds[0], marker='_', c='r', s=40)
    plt.scatter(np.arange(0,26,1), oStds[1], marker='_', c='r', s=40)
    plt.scatter(np.arange(0,26,1), dMeans, s=10, color=dc)
    plt.scatter(np.arange(0,26,1), oMeans, s=10, color=nc)
    plt.plot(np.arange(0,26,1), dMeans, label=d2, color=dc)
    plt.plot(np.arange(0,26,1), oMeans, label=d1, color=nc)
    for i in range(0, len(dPairs)):
        plt.plot([i,i], dPairs[i], color=dc)
    for i in range(0, len(oPairs)):
        plt.plot([i,i], oPairs[i], color=nc)
    plt.xticks(ticks=np.arange(0,26,1), labels=all_included)
    plt.title(band+" PSD " + ("EC" if closed else "EO"), fontsize=18)
    plt.legend(loc='upper right')
    plt.xlabel("Channel", fontsize=16)
    plt.ylabel("PSD", fontsize=16)
    plt.show()
    
#for b in bands:
#    mdd_healthy_ranges(b, 'HEALTHY', 'MDD', closed=False)




##################################################################################################
##################################################################################################
##################################################################################################


def regions_means_multiple(band, num_disorders, closed, outliers):
    if(closed):
        all_psds = np.load('disorders_EC_psds.npy', allow_pickle=True)
    else:
        all_psds = np.load('disorders_EO_psds.npy', allow_pickle=True)

    index = bands.index(band)

    all = []

    counts = [0 for i in range(0, num_disorders)]

    for i in range(0,num_disorders):
        all.append(np.zeros(26))

    for ind in all_psds:
        psds = ind[1:]
        disorder = int(ind[0])-1
        total = np.zeros(26)
        for i in range(0,60):
            res = []
            for j in range(0,26):
                val = psds[(i*130)+j*5+index]
                if(outliers):
                    res.append(val)
                elif(disorder==0 and (j==10 or j==14) and (index==4)):
                    res.append(val if(val < 20) else 0)
                else:
                    res.append(val if(val <= 1000) else 0)
            res = np.array(res)
            total = total+res
        total = total/60

        disorder = int(ind[0])-1
        if(disorder < num_disorders and disorder >= 0):
            counts[disorder] += 1
            all[disorder] = all[disorder] + total

    regions_means = []
    for i in range(0, num_disorders):
        all[i] = all[i]/counts[i]
        F = np.array([all[i][k] for k in frontal])
        T = np.array([all[i][k] for k in temporal])
        C = np.array([all[i][k] for k in central])
        P = np.array([all[i][k] for k in parietal])
        O = np.array([all[i][k] for k in occupital])

        regions_means.append([np.mean(F), np.mean(T), np.mean(C), np.mean(P), np.mean(O)])

    mx = 0
    mn = 200

    for r in regions_means:
        if(max(r) > mx):
            mx = max(r)
        if(min(r) < mn):
            mn = min(r)

    scaled = []
    for r in regions_means:
        vals = []
        for item in r:
            vals.append((item-mn)/(mx-mn))
        scaled.append(vals)

    scaled = np.array(scaled)

    """
    i=0
    for s in scaled:
        plt.plot(s, label=diagnoses[i+1])
        plt.scatter(['Frontal', 'Temporal', 'Central', 'Parietal', 'Occupital'], vals, s=10)
        i+=1

    plt.title("Mean " +band+" PSD " + ("(EC)" if closed else "(EO)"), fontsize=18)
    plt.xticks(ticks=np.arange(0,5,1), labels=['Frontal', 'Temporal', 'Central', 'Parietal', 'Occupital'])
    plt.yticks(ticks=np.arange(0,1.01,.1))
    plt.legend(bbox_to_anchor=(1.0, 1.0), loc='upper left')
    plt.xlabel("Region", fontsize=14)
    plt.ylabel("PSD (scaled)", fontsize=14)
    plt.tight_layout()
    plt.show()
    """
    return scaled

#for b in bands:
#   regions_means_multiple(b, 7, closed=False, outliers=False)


def plot_PSD_Comparison():

    figure, axis = plt.subplots(3, 2, figsize=(17, 17)) 
  
    figure.suptitle("Mean PSD Values for Regions of Brain", fontsize=24, weight="bold")

    # For DELTA EC
    delta_EC = regions_means_multiple("Delta", 7, closed=True, outliers=False)
    i=0
    for vals in delta_EC:
        axis[0, 0].plot(vals, label=diagnoses[i+1])
        axis[0, 0].scatter(['Frontal', 'Temporal', 'Central', 'Parietal', 'Occupital'], vals, s=10)
        i+=1

    axis[0, 0].set_title("(a) Delta PSD (EC)", fontsize=20, weight="bold")
    axis[0, 0].set_xticks(ticks=np.arange(0,5,1), labels=['Frontal', 'Temporal', 'Central', 'Parietal', 'Occupital'], fontsize=14)
    axis[0, 0].set_yticks(ticks=np.arange(0,1.01,.1), labels=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],fontsize=14)
    axis[0, 0].set_xlabel("Region", fontsize=18)
    axis[0, 0].set_ylabel("PSD (scaled)", fontsize=18)
    
    # For DELTA EO
    delta_EO = regions_means_multiple("Delta", 7, closed=False, outliers=False)
    i=0
    for vals in delta_EO:
        axis[0, 1].plot(vals)
        axis[0, 1].scatter(['Frontal', 'Temporal', 'Central', 'Parietal', 'Occupital'], vals, s=10)
        i+=1

    axis[0, 1].set_title("(b) Delta PSD (EO)", fontsize=20, weight="bold")
    axis[0, 1].set_xticks(ticks=np.arange(0,5,1), labels=['Frontal', 'Temporal', 'Central', 'Parietal', 'Occupital'], fontsize=14)
    axis[0, 1].set_yticks(ticks=np.arange(0,1.01,.1), labels=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],fontsize=14)
    axis[0, 1].set_xlabel("Region", fontsize=18)
    axis[0, 1].set_ylabel("PSD (scaled)", fontsize=18)

    # For THETA EC
    theta_EC = regions_means_multiple("Theta", 7, closed=True, outliers=False)
    i=0
    for vals in theta_EC:
        axis[1, 0].plot(vals)
        axis[1, 0].scatter(['Frontal', 'Temporal', 'Central', 'Parietal', 'Occupital'], vals, s=10)
        i+=1

    axis[1, 0].set_title("(c) Theta PSD (EC)", fontsize=20, weight="bold")
    axis[1, 0].set_xticks(ticks=np.arange(0,5,1), labels=['Frontal', 'Temporal', 'Central', 'Parietal', 'Occupital'], fontsize=14)
    axis[1, 0].set_yticks(ticks=np.arange(0,1.01,.1), labels=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],fontsize=14)
    axis[1, 0].set_xlabel("Region", fontsize=18)
    axis[1, 0].set_ylabel("PSD (scaled)", fontsize=18)

    # For THETA EO
    theta_EO = regions_means_multiple("Theta", 7, closed=False, outliers=False)
    i=0
    for vals in theta_EO:
        axis[1, 1].plot(vals)
        axis[1, 1].scatter(['Frontal', 'Temporal', 'Central', 'Parietal', 'Occupital'], vals, s=10)
        i+=1

    axis[1, 1].set_title("(d) Theta PSD (EO)", fontsize=20, weight="bold")
    axis[1, 1].set_xticks(ticks=np.arange(0,5,1), labels=['Frontal', 'Temporal', 'Central', 'Parietal', 'Occupital'], fontsize=14)
    axis[1, 1].set_yticks(ticks=np.arange(0,1.01,.1), labels=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],fontsize=14)
    axis[1, 1].set_xlabel("Region", fontsize=18)
    axis[1, 1].set_ylabel("PSD (scaled)", fontsize=18)

    # For Alpha EC
    alpha_EC = regions_means_multiple("Alpha", 7, closed=True, outliers=False)
    i=0
    for vals in alpha_EC:
        axis[2, 0].plot(vals)
        axis[2, 0].scatter(['Frontal', 'Temporal', 'Central', 'Parietal', 'Occupital'], vals, s=10)
        i+=1

    axis[2, 0].set_title("(e) Alpha PSD (EC)", fontsize=20, weight="bold")
    axis[2, 0].set_xticks(ticks=np.arange(0,5,1), labels=['Frontal', 'Temporal', 'Central', 'Parietal', 'Occupital'], fontsize=14)
    axis[2, 0].set_yticks(ticks=np.arange(0,1.01,.1), labels=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],fontsize=14)
    axis[2, 0].set_xlabel("Region", fontsize=18)
    axis[2, 0].set_ylabel("PSD (scaled)", fontsize=18)

    # For Alpha EO
    alpha_EO = regions_means_multiple("Alpha", 7, closed=False, outliers=False)
    i=0
    for vals in alpha_EO:
        axis[2, 1].plot(vals)
        axis[2, 1].scatter(['Frontal', 'Temporal', 'Central', 'Parietal', 'Occupital'], vals, s=10)
        i+=1

    axis[2, 1].set_title("(f) Alpha PSD (EO)", fontsize=20, weight="bold")
    axis[2, 1].set_xticks(ticks=np.arange(0,5,1), labels=['Frontal', 'Temporal', 'Central', 'Parietal', 'Occupital'], fontsize=14)
    axis[2, 1].set_yticks(ticks=np.arange(0,1.01,.1), labels=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],fontsize=14)
    axis[2, 1].set_xlabel("Region", fontsize=18)
    axis[2, 1].set_ylabel("PSD (scaled)", fontsize=18)


    figure.legend(loc="lower center", fontsize=16, ncol=7)
    plt.subplots_adjust(hspace = 0.3, wspace=0.2)
    #plt.tight_layout()
    plt.savefig("psd_Comparison_test")

#plot_PSD_Comparison()

##################################################################################################
##################################################################################################
##################################################################################################
    
def mdd_healthy_comp():
    ec_psds = np.load('disorders_EC_psds.npy', allow_pickle=True)

    d_count = 0
    depressed = []
    for i in range(0, 26):
        depressed.append([])

    o_count = 0
    other = []
    for i in range(0, 26):
        other.append([])

    for ind in ec_psds:
        psds = ind[1:]
        total = []
        for i in range(0, 26):
            total.append([])
        for i in range(0,60):
            for j in range(0,26):
                total[j].append(psds[(i*130)+j*5+2])

        if(ind[0] == 4.0):
            o_count += 1
            for i in range(0, 26):
                for j in total[i]:
                    other[i].append(j)
        elif(ind[0] == 2.0):
            d_count += 1
            for i in range(0, 26):
                for j in total[i]:
                    depressed[i].append(j)
    

    #, showfliers=False
    x_pos_range = np.arange(2)
    x_pos = (x_pos_range * 0.5) + 0.75

    bp1=plt.boxplot(
        depressed, sym='', widths=0.3, labels=all_included, patch_artist=True,notch=True,showmeans=True,
        positions=[x_pos[0] + j*1 for j in range(0, 26)]
    )
    bp2=plt.boxplot(
        other, sym='', widths=0.3,patch_artist=True, notch=True,showmeans=True,
        positions=[x_pos[1] + j*1 for j in range(0, 26)]
    )

    for box in bp1['boxes']:
        box.set_facecolor("aquamarine")

    plt.title("Delta PSD EO", fontsize=20)
    plt.xticks(ticks=np.arange(1,27,1), labels=all_included)
    plt.legend([bp1["boxes"][0], bp2["boxes"][0]], ['Depressed', 'Healthy'], loc='upper right')
    plt.xlabel("Channel", fontsize=16)
    plt.ylabel("PSD", fontsize=16)
    plt.show()

def normal_vs_mental_ec():
    accs_normal = np.load('EC_ONLY_ACCS.npy', allow_pickle=True)
    accs_mental = np.load('diff_MENTAL_EC_ACCS.npy', allow_pickle=True)

    labels = np.arange(0, 1000, 1)
    plt.figure(figsize=(10,5))
    plt.plot(accs_mental, label="MENTAL")
    plt.plot(accs_normal, label="EEG Only")
    plt.title("Accuracy of MENTAL Compared to EEG Only (EC)", fontsize=16)
    plt.ylabel("Accuracy", fontsize=12)
    plt.xlabel("Epoch", fontsize=12)
    plt.yticks(ticks=np.arange(0,1.01,0.1))
    plt.xticks(ticks=np.arange(0,1001,100))
    plt.legend(loc="upper right")

    plt.savefig("mental_vs_eeg_ec")
    plt.clf()

def normal_vs_mental_eo():
    accs_normal = np.load('EO_ONLY_ACCS.npy', allow_pickle=True)
    accs_mental = np.load('diff_MENTAL_EO_ACCS.npy', allow_pickle=True)

    labels = np.arange(0, 1000, 1)
    plt.figure(figsize=(10,5))
    plt.plot(accs_mental, label="MENTAL")
    plt.plot(accs_normal, label="EEG Only")
    plt.title("Accuracy of MENTAL Compared to EEG Only (EO)", fontsize=16)
    plt.ylabel("Accuracy", fontsize=12)
    plt.xlabel("Epoch", fontsize=12)
    plt.yticks(ticks=np.arange(0,1.01,0.1))
    plt.xticks(ticks=np.arange(0,1001,100))
    plt.legend(loc="upper right")

    plt.savefig("mental_vs_eeg_eo")
    plt.clf()

def normal_vs_mental_ec_eo():
    accs_normal = np.load("C:\\Users\\glgre\\Documents\\ResearchCode\\EO_EC_ACCS.npy", allow_pickle=True)
    accs_mental = np.load('diff_MENTAL_EC_EO_ACCS.npy', allow_pickle=True)

    labels = np.arange(0, 1000, 1)
    plt.figure(figsize=(10,5))
    plt.plot(accs_mental, label="MENTAL")
    plt.plot(accs_normal, label="EEG Only")
    plt.title("Accuracy of MENTAL Compared to EEG Only (EC+EO)", fontsize=16)
    plt.ylabel("Accuracy", fontsize=12)
    plt.xlabel("Epoch", fontsize=12)
    plt.yticks(ticks=np.arange(0,1.01,0.1))
    plt.xticks(ticks=np.arange(0,1001,100))
    plt.legend(loc="upper right")

    plt.savefig("mental_vs_eeg_ec_eo")
    plt.clf()

def normal_vs_adhd_mental_ec():
    accs_normal = np.load("C:\\Users\\glgre\\Documents\\ResearchCode\\EC_ONLY_ADHD_ACCS.npy", allow_pickle=True)
    accs_mental = np.load("C:\\Users\\glgre\\Documents\\ResearchCode\\ADHD_MENTAL_EC_ACCS.npy", allow_pickle=True)

    labels = np.arange(0, 1000, 1)
    plt.figure(figsize=(10,5))
    plt.plot(accs_mental, label="MENTAL")
    plt.plot(accs_normal, label="EEG Only")
    plt.title("Accuracy of ADHD MENTAL Compared to EEG Only (EC)", fontsize=16)
    plt.ylabel("Accuracy", fontsize=12)
    plt.xlabel("Epoch", fontsize=12)
    plt.yticks(ticks=np.arange(0,1.01,0.1))
    plt.xticks(ticks=np.arange(0,1001,100))
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.show()
    #plt.savefig("adhd_mental_vs_eeg_ec")
    #plt.clf()

def normal_vs_adhd_mental_eo():
    accs_normal = np.load("C:\\Users\\glgre\\Documents\\ResearchCode\\EO_ONLY_ADHD_ACCS.npy", allow_pickle=True)
    accs_mental = np.load("C:\\Users\\glgre\\Documents\\ResearchCode\\ADHD_MENTAL_EO_ACCS.npy", allow_pickle=True)

    labels = np.arange(0, 1000, 1)
    plt.figure(figsize=(10,5))
    plt.plot(accs_mental, label="MENTAL")
    plt.plot(accs_normal, label="EEG Only")
    plt.title("Accuracy of ADHD MENTAL Compared to EEG Only (EO)", fontsize=16)
    plt.ylabel("Accuracy", fontsize=12)
    plt.xlabel("Epoch", fontsize=12)
    plt.yticks(ticks=np.arange(0,1.01,0.1))
    plt.xticks(ticks=np.arange(0,1001,100))
    plt.legend(loc="upper right")

    plt.savefig("adhd_mental_vs_eeg_eo")
    plt.clf()

def normal_vs_adhd_mental_ec_eo():
    accs_normal = np.load("C:\\Users\\glgre\\Documents\\ResearchCode\\EO_EC_ADHD_ACCS.npy", allow_pickle=True)
    accs_mental = np.load("C:\\Users\\glgre\\Documents\\ResearchCode\\ADHD_MENTAL_EC_EO_ACCS.npy", allow_pickle=True)

    labels = np.arange(0, 1000, 1)
    plt.figure(figsize=(10,5))
    plt.plot(accs_mental, label="MENTAL")
    plt.plot(accs_normal, label="EEG Only")
    plt.title("Accuracy of ADHD MENTAL Compared to EEG Only (EC+EO)", fontsize=16)
    plt.ylabel("Accuracy", fontsize=12)
    plt.xlabel("Epoch", fontsize=12)
    plt.yticks(ticks=np.arange(0,1.01,0.1))
    plt.xticks(ticks=np.arange(0,1001,100))
    plt.legend(loc="upper right")

    plt.savefig("adhd_mental_vs_eeg_ec_eo")
    plt.clf() 

def plot_MENTAL_EEG_Comparison():

    mc = "tab:orange"
    nc = "tab:blue"

    figure, axis = plt.subplots(3, 2, figsize=(15, 14)) 
  
    figure.suptitle("Accuracy of MENTAL vs. EEG Only (1 Disorder vs All Disorders)", fontsize=24, weight="bold")

    # For ADHD EO
    adhd_accs_normal = np.load("C:\\Users\\glgre\\Documents\\ResearchCode\\EC_ONLY_ADHD_ACCS.npy", allow_pickle=True)
    adhd_accs_mental = np.load("C:\\Users\\glgre\\Documents\\ResearchCode\\ADHD_MENTAL_EC_ACCS.npy", allow_pickle=True)
    axis[0, 0].plot(adhd_accs_normal, label="EEG Only", c= nc)
    axis[0, 0].plot(adhd_accs_mental, label="MENTAL", c= mc) 
    axis[0, 0].set_xlabel("Epoch", fontsize=16)
    axis[0, 0].set_ylabel("Accuracy", fontsize=16)
    axis[0, 0].set_xticks(ticks=np.arange(0,1001,100), labels=np.arange(0,1001,100), fontsize=14)
    axis[0, 0].set_yticks(ticks=np.arange(0,1.01,0.1), labels=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1], fontsize=14)
    axis[0, 0].set_title("(a) ADHD (EC)", fontsize=16, weight="bold") 
    
    # For MDD EC
    mdd_accs_normal = np.load('EC_ONLY_ACCS.npy', allow_pickle=True)
    mdd_accs_mental = np.load('diff_MENTAL_EC_ACCS.npy', allow_pickle=True)
    axis[0, 1].plot(mdd_accs_normal, c= nc)
    axis[0, 1].plot(mdd_accs_mental, c= mc) 
    axis[0, 1].set_xlabel("Epoch", fontsize=16)
    axis[0, 1].set_ylabel("Accuracy", fontsize=16)
    axis[0, 1].set_xticks(ticks=np.arange(0,1001,100), labels=np.arange(0,1001,100), fontsize=14)
    axis[0, 1].set_yticks(ticks=np.arange(0,1.01,0.1), labels=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1], fontsize=14)
    axis[0, 1].set_title("(b) MDD (EC)", fontsize=16, weight="bold") 

    # For ADHD EO
    adhd_accs_normal = np.load("C:\\Users\\glgre\\Documents\\ResearchCode\\EO_ONLY_ADHD_ACCS.npy", allow_pickle=True)
    adhd_accs_mental = np.load("C:\\Users\\glgre\\Documents\\ResearchCode\\ADHD_MENTAL_EO_ACCS.npy", allow_pickle=True)
    axis[1, 0].plot(adhd_accs_normal, c= nc)
    axis[1, 0].plot(adhd_accs_mental, c= mc) 
    axis[1, 0].set_xlabel("Epoch", fontsize=16)
    axis[1, 0].set_ylabel("Accuracy", fontsize=16)
    axis[1, 0].set_xticks(ticks=np.arange(0,1001,100), labels=np.arange(0,1001,100), fontsize=14)
    axis[1, 0].set_yticks(ticks=np.arange(0,1.01,0.1), labels=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1], fontsize=14)
    axis[1, 0].set_title("(c) ADHD (EO)", fontsize=16, weight="bold") 
    
    # For MDD EO
    mdd_accs_normal = np.load('EO_ONLY_ACCS.npy', allow_pickle=True)
    mdd_accs_mental = np.load('diff_MENTAL_EO_ACCS.npy', allow_pickle=True)
    axis[1, 1].plot(mdd_accs_normal, c= nc)
    axis[1, 1].plot(mdd_accs_mental, c= mc) 
    axis[1, 1].set_xlabel("Epoch", fontsize=16)
    axis[1, 1].set_ylabel("Accuracy", fontsize=16)
    axis[1, 1].set_xticks(ticks=np.arange(0,1001,100), labels=np.arange(0,1001,100), fontsize=14)
    axis[1, 1].set_yticks(ticks=np.arange(0,1.01,0.1), labels=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1], fontsize=14)
    axis[1, 1].set_title("(d) MDD (EO)", fontsize=16, weight="bold") 

    # For ADHD EC+EO
    adhd_accs_normal = np.load("C:\\Users\\glgre\\Documents\\ResearchCode\\EO_EC_ADHD_ACCS.npy", allow_pickle=True)
    adhd_accs_mental = np.load("C:\\Users\\glgre\\Documents\\ResearchCode\\ADHD_MENTAL_EC_EO_ACCS.npy", allow_pickle=True)
    axis[2, 0].plot(adhd_accs_normal, c= nc)
    axis[2, 0].plot(adhd_accs_mental, c= mc)
    axis[2, 0].set_xlabel("Epoch", fontsize=16)
    axis[2, 0].set_ylabel("Accuracy", fontsize=16)
    axis[2, 0].set_xticks(ticks=np.arange(0,1001,100), labels=np.arange(0,1001,100), fontsize=14)
    axis[2, 0].set_yticks(ticks=np.arange(0,1.01,0.1), labels=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1], fontsize=14)
    axis[2, 0].set_title("(e) ADHD (EC+EO)", fontsize=16, weight="bold") 
    
    # For MDD EC+EO
    mdd_accs_normal = np.load("C:\\Users\\glgre\\Documents\\ResearchCode\\EO_EC_ACCS.npy", allow_pickle=True)
    mdd_accs_mental = np.load('diff_MENTAL_EC_EO_ACCS.npy', allow_pickle=True)
    axis[2, 1].plot(mdd_accs_normal, c= nc)
    axis[2, 1].plot(mdd_accs_mental, c= mc) 
    axis[2, 1].set_xlabel("Epoch", fontsize=16)
    axis[2, 1].set_ylabel("Accuracy", fontsize=16)
    axis[2, 1].set_xticks(ticks=np.arange(0,1001,100), labels=np.arange(0,1001,100), fontsize=14)
    axis[2, 1].set_yticks(ticks=np.arange(0,1.01,0.1), labels=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1], fontsize=14)
    axis[2, 1].set_title("(f) MDD (EC+EO)", fontsize=16, weight="bold") 

    figure.legend(loc="lower center", fontsize=20)
    plt.subplots_adjust(hspace = 0.3, wspace=0.15)
    #plt.tight_layout()
    plt.savefig("testerrr")
    
#plot_MENTAL_EEG_Comparison()

def plot_MENTAL_EEG_Control_Comparison():

    mc = "tab:orange"
    nc = "tab:blue"

    figure, axis = plt.subplots(3, 2, figsize=(15, 14)) 
  
    figure.suptitle("Accuracy of MENTAL vs. EEG Only (One Disorder vs Healthy)", fontsize=24, weight="bold")

    # For ADHD EC
    adhd_accs_normal = np.load("C:\\Users\\glgre\\Documents\\ResearchCode\\controls\\EC_ONLY_ADHD_HEALTHY_ACCS.npy", allow_pickle=True)
    adhd_accs_mental = np.load("C:\\Users\\glgre\\Documents\\ResearchCode\\mentalControls\\MENTAL_EC_HEALTHY_ADHD_ACCS.npy", allow_pickle=True)
    axis[0, 0].plot(adhd_accs_normal, label="EEG Only", c= nc)
    axis[0, 0].plot(adhd_accs_mental, label="MENTAL", c= mc) 
    axis[0, 0].set_xlabel("Epoch", fontsize=16)
    axis[0, 0].set_ylabel("Accuracy", fontsize=16)
    axis[0, 0].set_xticks(ticks=np.arange(0,501,100), labels=np.arange(0,501,100), fontsize=14)
    axis[0, 0].set_yticks(ticks=np.arange(0,1.01,0.1), labels=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1], fontsize=14)
    axis[0, 0].set_title("(a) ADHD (EC)", fontsize=16, weight="bold") 
    print("\n\nADHD EC:")
    print(f"\tMENTAL: {adhd_accs_mental[499]}")
    print(f"\tNormal: {adhd_accs_normal[499]}")

    # For MDD EC
    mdd_accs_normal = np.load("C:\\Users\\glgre\\Documents\\ResearchCode\\controls\\EC_ONLY_MDD_HEALTHY_ACCS.npy", allow_pickle=True)
    mdd_accs_mental = np.load("C:\\Users\\glgre\\Documents\\ResearchCode\\mentalControls\\MENTAL_EC_HEALTHY_MDD_ACCS.npy", allow_pickle=True)
    axis[0, 1].plot(mdd_accs_normal, c= nc)
    axis[0, 1].plot(mdd_accs_mental, c= mc) 
    axis[0, 1].set_xlabel("Epoch", fontsize=16)
    axis[0, 1].set_ylabel("Accuracy", fontsize=16)
    axis[0, 1].set_xticks(ticks=np.arange(0,501,100), labels=np.arange(0,501,100), fontsize=14)
    axis[0, 1].set_yticks(ticks=np.arange(0,1.01,0.1), labels=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1], fontsize=14)
    axis[0, 1].set_title("(b) MDD (EC)", fontsize=16, weight="bold") 
    print("\n\nMDD EC:")
    print(f"\tMENTAL: {mdd_accs_mental[499]}")
    print(f"\tNormal: {mdd_accs_normal[499]}")

    # For ADHD EO
    adhd_accs_normal = np.load("C:\\Users\\glgre\\Documents\\ResearchCode\\controls\\EO_ONLY_ADHD_HEALTHY_ACCS.npy", allow_pickle=True)
    adhd_accs_mental = np.load("C:\\Users\\glgre\\Documents\\ResearchCode\\mentalControls\\MENTAL_EO_HEALTHY_ADHD_ACCS.npy", allow_pickle=True)
    axis[1, 0].plot(adhd_accs_normal, c= nc)
    axis[1, 0].plot(adhd_accs_mental, c= mc) 
    axis[1, 0].set_xlabel("Epoch", fontsize=16)
    axis[1, 0].set_ylabel("Accuracy", fontsize=16)
    axis[1, 0].set_xticks(ticks=np.arange(0,501,100), labels=np.arange(0,501,100), fontsize=14)
    axis[1, 0].set_yticks(ticks=np.arange(0,1.01,0.1), labels=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1], fontsize=14)
    axis[1, 0].set_title("(c) ADHD (EO)", fontsize=16, weight="bold") 
    print("\n\nADHD EO:")
    print(f"\tMENTAL: {adhd_accs_mental[499]}")
    print(f"\tNormal: {adhd_accs_normal[499]}")
    
    # For MDD EO
    mdd_accs_normal = np.load("C:\\Users\\glgre\\Documents\\ResearchCode\\controls\\EO_ONLY_MDD_HEALTHY_ACCS.npy", allow_pickle=True)
    mdd_accs_mental = np.load("C:\\Users\\glgre\\Documents\\ResearchCode\\mentalControls\\MENTAL_EO_HEALTHY_MDD_ACCS.npy", allow_pickle=True)
    axis[1, 1].plot(mdd_accs_normal, c= nc)
    axis[1, 1].plot(mdd_accs_mental, c= mc) 
    axis[1, 1].set_xlabel("Epoch", fontsize=16)
    axis[1, 1].set_ylabel("Accuracy", fontsize=16)
    axis[1, 1].set_xticks(ticks=np.arange(0,501,100), labels=np.arange(0,501,100), fontsize=14)
    axis[1, 1].set_yticks(ticks=np.arange(0,1.01,0.1), labels=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1], fontsize=14)
    axis[1, 1].set_title("(d) MDD (EO)", fontsize=16, weight="bold") 
    print("\n\nMDD EO:")
    print(f"\tMENTAL: {mdd_accs_mental[499]}")
    print(f"\tNormal: {mdd_accs_normal[499]}")

    # For ADHD EC+EO
    adhd_accs_normal = np.load("C:\\Users\\glgre\\Documents\\ResearchCode\\controls\\EO_EC_ADHD_HEALTHY_ACCS.npy", allow_pickle=True)
    adhd_accs_mental = np.load("C:\\Users\\glgre\\Documents\\ResearchCode\\mentalControls\\MENTAL_EC_EO_HEALTHY_ADHD_ACCS.npy", allow_pickle=True)
    axis[2, 0].plot(adhd_accs_normal, c= nc)
    axis[2, 0].plot(adhd_accs_mental, c= mc)
    axis[2, 0].set_xlabel("Epoch", fontsize=16)
    axis[2, 0].set_ylabel("Accuracy", fontsize=16)
    axis[2, 0].set_xticks(ticks=np.arange(0,501,100), labels=np.arange(0,501,100), fontsize=14)
    axis[2, 0].set_yticks(ticks=np.arange(0,1.01,0.1), labels=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1], fontsize=14)
    axis[2, 0].set_title("(e) ADHD (EC+EO)", fontsize=16, weight="bold") 
    print("\n\nADHD EC+EO:")
    print(f"\tMENTAL: {adhd_accs_mental[499]}")
    print(f"\tNormal: {adhd_accs_normal[499]}")
    
    # For MDD EC+EO
    mdd_accs_normal = np.load("C:\\Users\\glgre\\Documents\\ResearchCode\\controls\\EO_EC_MDD_HEALTHY_ACCS.npy", allow_pickle=True)
    mdd_accs_mental = np.load("C:\\Users\\glgre\\Documents\\ResearchCode\\mentalControls\\MENTAL_EC_EO_HEALTHY_MDD_ACCS.npy", allow_pickle=True)
    axis[2, 1].plot(mdd_accs_normal, c= nc)
    axis[2, 1].plot(mdd_accs_mental, c= mc) 
    axis[2, 1].set_xlabel("Epoch", fontsize=16)
    axis[2, 1].set_ylabel("Accuracy", fontsize=16)
    axis[2, 1].set_xticks(ticks=np.arange(0,501,100), labels=np.arange(0,501,100), fontsize=14)
    axis[2, 1].set_yticks(ticks=np.arange(0,1.01,0.1), labels=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1], fontsize=14)
    axis[2, 1].set_title("(f) MDD (EC+EO)", fontsize=16, weight="bold") 
    print("\n\nMDD EC+EO:")
    print(f"\tMENTAL: {mdd_accs_mental[499]}")
    print(f"\tNormal: {mdd_accs_normal[499]}")

    figure.legend(loc="lower center", fontsize=20)
    plt.subplots_adjust(hspace = 0.3, wspace=0.15)
    #plt.tight_layout()
    plt.savefig("mental_eeg_control_comparison_new")
    
#plot_MENTAL_EEG_Control_Comparison()

def plot_accs():
    accs = np.load("C:\\Users\\glgre\\Documents\\ResearchCode\\controls\\EO_EC_ADHD_HEALTHY_ACCS.npy", allow_pickle=True)

    labels = np.arange(0, 500, 1)

    plt.figure(figsize=(15,10))
    plt.plot(labels, accs)
    plt.title("Accuracy of MDD EO for " + str(500) + " epochs")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.yticks(ticks=np.arange(0,1.01,0.1))

    plt.savefig("test")
    plt.show()



def plot_accs2():
    accs = np.load("C:\\Users\\glgre\\Documents\\ResearchCode\\MDD_HEALTH_MENTAL_EC_ACCS_epoch_999.npy", allow_pickle=True)

    labels = np.arange(0, 1000, 1)

    plt.figure(figsize=(15,10))
    plt.plot(labels, accs)
    plt.title("Accuracy of ADHD EC MENTAL for " + str(999+1) + " epochs")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.yticks(ticks=np.arange(0,1.01,0.1))

    plt.savefig("test")
    plt.show()


def imputed_adhd_mental_ec():
    accs = np.load("C:\\Users\\glgre\\Documents\\ResearchCode\\MENTAL_EO_IMPUTED_ADHD_ACCS.npy", allow_pickle=True)
    accs_normal = np.load("C:\\Users\\glgre\\Documents\\ResearchCode\\EO_ONLY_ADHD_ACCS.npy", allow_pickle=True)
    accs_mental = np.load("C:\\Users\\glgre\\Documents\\ResearchCode\\ADHD_MENTAL_EO_ACCS.npy", allow_pickle=True)

    labels = np.arange(0, 1000, 1)
    plt.figure(figsize=(10,5))
    plt.plot(accs_mental, label="MENTAL")
    plt.plot(accs_normal, label="EEG Only")
    plt.plot(accs, label="MENTAL with Imputed")
    plt.title("Accuracy of ADHD MENTAL Compared to EEG Only (EC)", fontsize=16)
    plt.ylabel("Accuracy", fontsize=12)
    plt.xlabel("Epoch", fontsize=12)
    plt.yticks(ticks=np.arange(0,1.01,0.1))
    plt.xticks(ticks=np.arange(0,1001,100))
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()

def imputed_mdd_mental():
    accs = np.load("C:\\Users\\glgre\\Documents\\ResearchCode\\MENTAL_EC_IMPUTED_MDD_ACCS.npy", allow_pickle=True)
    accs_normal = np.load('EC_ONLY_ACCS.npy', allow_pickle=True)
    accs_mental = np.load('diff_MENTAL_EC_ACCS.npy', allow_pickle=True)

    labels = np.arange(0, 1000, 1)
    plt.figure(figsize=(10,5))
    plt.plot(accs_mental, label="MENTAL")
    plt.plot(accs_normal, label="EEG Only")
    plt.plot(accs, label="MENTAL with Imputed")
    plt.title("Accuracy of MDD MENTAL Compared to EEG Only (EC)", fontsize=16)
    plt.ylabel("Accuracy", fontsize=12)
    plt.xlabel("Epoch", fontsize=12)
    plt.yticks(ticks=np.arange(0,1.01,0.1))
    plt.xticks(ticks=np.arange(0,1001,100))
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()


def plot_MENTAL_EEG_Imputed_Comparison():

    mc = "tab:orange"
    nc = "tab:blue"
    ic = "tab:orange"

    figure, axis = plt.subplots(3, 2, figsize=(15, 14)) 
  
    figure.suptitle("Accuracy of MENTAL vs. EEG Only (One Disorder vs All Disorders)", fontsize=24, weight="bold")

    # For ADHD EC
    adhd_accs_normal = np.load("C:\\Users\\glgre\\Documents\\ResearchCode\\EC_ONLY_ADHD_ACCS.npy", allow_pickle=True)
    adhd_accs_mental = np.load("C:\\Users\\glgre\\Documents\\ResearchCode\\ADHD_MENTAL_EC_ACCS.npy", allow_pickle=True)
    adhd_accs_impute = np.load("C:\\Users\\glgre\\Documents\\ResearchCode\\MENTAL_EC_IMPUTED_ADHD_ACCS.npy", allow_pickle=True)
    axis[0, 0].plot(adhd_accs_normal, label="EEG Only", c=nc)
    #axis[0, 0].plot(adhd_accs_mental, label="MENTAL"  , c=mc)
    axis[0, 0].plot(adhd_accs_impute, label="MENTAL" , c=ic) 
    axis[0, 0].set_xlabel("Epoch", fontsize=16)
    axis[0, 0].set_ylabel("Accuracy", fontsize=16)
    axis[0, 0].set_xticks(ticks=np.arange(0,1001,100), labels=np.arange(0,1001,100), fontsize=14)
    axis[0, 0].set_yticks(ticks=np.arange(0,1.01,0.1), labels=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1], fontsize=14)
    axis[0, 0].set_title("(a) ADHD (EC)", fontsize=16, weight="bold") 
    print("\n\nADHD EC:")
    print(f"\tMENTAL: {adhd_accs_impute[999]}")
    print(f"\tNormal: {adhd_accs_normal[999]}")
    
    # For MDD EC
    mdd_accs_normal = np.load('EC_ONLY_ACCS.npy', allow_pickle=True)
    mdd_accs_mental = np.load('diff_MENTAL_EC_ACCS.npy', allow_pickle=True)
    mdd_accs_impute = np.load("C:\\Users\\glgre\\Documents\\ResearchCode\\MENTAL_EC_IMPUTED_MDD_ACCS.npy", allow_pickle=True)
    axis[0, 1].plot(mdd_accs_normal, c=nc)
    #axis[0, 1].plot(mdd_accs_mental, c=mc) 
    axis[0, 1].plot(mdd_accs_impute, c=ic) 
    axis[0, 1].set_xlabel("Epoch", fontsize=16)
    axis[0, 1].set_ylabel("Accuracy", fontsize=16)
    axis[0, 1].set_xticks(ticks=np.arange(0,1001,100), labels=np.arange(0,1001,100), fontsize=14)
    axis[0, 1].set_yticks(ticks=np.arange(0,1.01,0.1), labels=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1], fontsize=14)
    axis[0, 1].set_title("(b) MDD (EC)", fontsize=16, weight="bold") 
    print("\n\nMDD EC:")
    print(f"\tMENTAL: {mdd_accs_impute[999]}")
    print(f"\tNormal: {mdd_accs_normal[999]}")


    # For ADHD EO
    adhd_accs_normal = np.load("C:\\Users\\glgre\\Documents\\ResearchCode\\EO_ONLY_ADHD_ACCS.npy", allow_pickle=True)
    adhd_accs_mental = np.load("C:\\Users\\glgre\\Documents\\ResearchCode\\ADHD_MENTAL_EO_ACCS.npy", allow_pickle=True)
    adhd_accs_impute = np.load("C:\\Users\\glgre\\Documents\\ResearchCode\\MENTAL_EO_IMPUTED_ADHD_ACCS.npy", allow_pickle=True)
    axis[1, 0].plot(adhd_accs_normal, c= nc)
    #axis[1, 0].plot(adhd_accs_mental, c= mc)
    axis[1, 0].plot(adhd_accs_impute, c= ic) 
    axis[1, 0].set_xlabel("Epoch", fontsize=16)
    axis[1, 0].set_ylabel("Accuracy", fontsize=16)
    axis[1, 0].set_xticks(ticks=np.arange(0,1001,100), labels=np.arange(0,1001,100), fontsize=14)
    axis[1, 0].set_yticks(ticks=np.arange(0,1.01,0.1), labels=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1], fontsize=14)
    axis[1, 0].set_title("(c) ADHD (EO)", fontsize=16, weight="bold") 
    print("\n\nADHD EO:")
    print(f"\tMENTAL: {adhd_accs_impute[999]}")
    print(f"\tNormal: {adhd_accs_normal[999]}")
    
    # For MDD EO
    mdd_accs_normal = np.load('EO_ONLY_ACCS.npy', allow_pickle=True)
    mdd_accs_mental = np.load('diff_MENTAL_EO_ACCS.npy', allow_pickle=True)
    mdd_accs_impute = np.load("C:\\Users\\glgre\\Documents\\ResearchCode\\MENTAL_EO_IMPUTED_MDD_ACCS.npy", allow_pickle=True)
    axis[1, 1].plot(mdd_accs_normal, c= nc)
    #axis[1, 1].plot(mdd_accs_mental, c= mc) 
    axis[1, 1].plot(mdd_accs_impute, c= ic) 
    axis[1, 1].set_xlabel("Epoch", fontsize=16)
    axis[1, 1].set_ylabel("Accuracy", fontsize=16)
    axis[1, 1].set_xticks(ticks=np.arange(0,1001,100), labels=np.arange(0,1001,100), fontsize=14)
    axis[1, 1].set_yticks(ticks=np.arange(0,1.01,0.1), labels=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1], fontsize=14)
    axis[1, 1].set_title("(d) MDD (EO)", fontsize=16, weight="bold")
    print("\n\nMDD EO:")
    print(f"\tMENTAL: {mdd_accs_impute[999]}")
    print(f"\tNormal: {mdd_accs_normal[999]}")

    # For ADHD EC+EO
    adhd_accs_normal = np.load("C:\\Users\\glgre\\Documents\\ResearchCode\\EO_EC_ADHD_ACCS.npy", allow_pickle=True)
    adhd_accs_mental = np.load("C:\\Users\\glgre\\Documents\\ResearchCode\\ADHD_MENTAL_EC_EO_ACCS.npy", allow_pickle=True)
    adhd_accs_impute = np.load("C:\\Users\\glgre\\Documents\\ResearchCode\\MENTAL_EC_EO_IMPUTED_ADHD_ACCS.npy", allow_pickle=True)
    axis[2, 0].plot(adhd_accs_normal, c= nc)
    #axis[2, 0].plot(adhd_accs_mental, c= mc)
    axis[2, 0].plot(adhd_accs_impute, c= ic)
    axis[2, 0].set_xlabel("Epoch", fontsize=16)
    axis[2, 0].set_ylabel("Accuracy", fontsize=16)
    axis[2, 0].set_xticks(ticks=np.arange(0,1001,100), labels=np.arange(0,1001,100), fontsize=14)
    axis[2, 0].set_yticks(ticks=np.arange(0,1.01,0.1), labels=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1], fontsize=14)
    axis[2, 0].set_title("(e) ADHD (EC+EO)", fontsize=16, weight="bold") 

    print("\n\nADHD EC+EO:")
    print(f"\tMENTAL: {adhd_accs_impute[999]}")
    print(f"\tNormal: {adhd_accs_normal[999]}")
    
    # For MDD EC+EO
    mdd_accs_normal = np.load("C:\\Users\\glgre\\Documents\\ResearchCode\\EO_EC_ACCS.npy", allow_pickle=True)
    #mdd_accs_mental = np.load('diff_MENTAL_EC_EO_ACCS.npy', allow_pickle=True)
    mdd_accs_impute = np.load("C:\\Users\\glgre\\Documents\\ResearchCode\\MENTAL_EC_EO_IMPUTED_MDD_ACCS.npy", allow_pickle=True)
    axis[2, 1].plot(mdd_accs_normal, c= nc)
    #axis[2, 1].plot(mdd_accs_mental, c= mc) 
    axis[2, 1].plot(mdd_accs_impute, c= ic)
    axis[2, 1].set_xlabel("Epoch", fontsize=16)
    axis[2, 1].set_ylabel("Accuracy", fontsize=16)
    axis[2, 1].set_xticks(ticks=np.arange(0,1001,100), labels=np.arange(0,1001,100), fontsize=14)
    axis[2, 1].set_yticks(ticks=np.arange(0,1.01,0.1), labels=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1], fontsize=14)
    axis[2, 1].set_title("(f) MDD (EC+EO)", fontsize=16, weight="bold") 

    figure.legend(loc="lower center", fontsize=20)
    plt.subplots_adjust(hspace = 0.3, wspace=0.15)

    print("\n\nMDD EC+EO:")
    print(f"\tMENTAL: {mdd_accs_impute[999]}")
    print(f"\tNormal: {mdd_accs_normal[999]}")
    #plt.show()
    plt.savefig("modelcomparison")
    
#plot_MENTAL_EEG_Imputed_Comparison()

def plot_MENTAL_MultiClass_Comparison():

    nc = "tab:blue"
    ic = "tab:orange"
    oc = "tab:green"

    figure, axis = plt.subplots(1, 2, figsize=(15, 5)) 
  
    figure.suptitle("Accuracy of MENTAL (MDD, ADHD, SMC, OCD, Healthy)", fontsize=24, weight="bold")

    # For EC Multi Class
    #e2_accs = np.load("C:\\Users\\glgre\\Documents\\ResearchCode\\TOP5_e2_MENTAL_EC_IMPUTED_ACCS.npy", allow_pickle=True)
    #e3_accs = np.load("C:\\Users\\glgre\\Documents\\ResearchCode\\e3e4Original\\TOP5_e4_MENTAL_EC_IMPUTED_ACCS.npy", allow_pickle=True)
    e4_accs = np.load("C:\\Users\\glgre\\Documents\\ResearchCode\\TOP5_e4_MENTAL_EC_IMPUTED_ACCS.npy", allow_pickle=True)
    #e5_accs = np.load("C:\\Users\\glgre\\Documents\\ResearchCode\\TOP5_e5_MENTAL_EC_IMPUTED_ACCS.npy", allow_pickle=True)
    #axis[0].plot(e2_accs, label="MENTAL e2", c=nc)
    #axis[0].plot(e3_accs, label="MENTAL", c=ic) 
    axis[0].plot(e4_accs[:1000], label="MENTAL", c=ic)
    #axis[0].plot(e5_accs, label="MENTAL e5", c=ic)
    axis[0].set_xlabel("Epoch", fontsize=16)
    axis[0].set_ylabel("Accuracy", fontsize=16)
    axis[0].set_xticks(ticks=np.arange(0,1001,100), labels=np.arange(0,1001,100), fontsize=14)
    axis[0].set_yticks(ticks=np.arange(0,1.01,0.1), labels=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1], fontsize=14)
    axis[0].set_title("(a) Four Disorder + Healthy MENTAL EC", fontsize=16, weight="bold") 

    print("\n\nMulti-class prediction EC:")
    #print(f"\tMENTAL e2: {e2_accs[999]}")
    #print(f"\tMENTAL e3: {e3_accs[999]}")
    print(f"\tMENTAL e4: {e4_accs[1999]}")
    #print(f"\tMENTAL e5: {e5_accs[1999]}\n")

    #print(f"\tmax MENTAL e2: {np.max(e2_accs)}")
    #print(f"\tmax MENTAL e3: {np.max(e3_accs)}")
    print(f"\tmax MENTAL e4: {np.max(e4_accs)}")
    print(np.where(e4_accs == 0.7272727272727273))
    #print(f"\tmax MENTAL e5: {np.max(e5_accs)}")

    # For EO Multi Class
    #e2_accs = np.load("C:\\Users\\glgre\\Documents\\ResearchCode\\TOP5_e2_MENTAL_EO_IMPUTED_ACCS.npy", allow_pickle=True)
    e3_accs = np.load("C:\\Users\\glgre\\Documents\\ResearchCode\\e3e4Original\\TOP5_e4_MENTAL_EO_IMPUTED_ACCS.npy", allow_pickle=True)
    #e4_accs = np.load("C:\\Users\\glgre\\Documents\\ResearchCode\\TOP5_e4_MENTAL_EO_IMPUTED_ACCS.npy", allow_pickle=True)
    #e5_accs = np.load("C:\\Users\\glgre\\Documents\\ResearchCode\\TOP5_e5_MENTAL_EO_IMPUTED_ACCS.npy", allow_pickle=True)
    #axis[1].plot(e2_accs, c=nc)
    axis[1].plot(e3_accs, c=ic) 
    #axis[1].plot(e4_accs, c=oc)
    #axis[1].plot(e5_accs, c=ic) 
    axis[1].set_xlabel("Epoch", fontsize=16)
    axis[1].set_ylabel("Accuracy", fontsize=16)
    axis[1].set_xticks(ticks=np.arange(0,1001,100), labels=np.arange(0,1001,100), fontsize=14)
    axis[1].set_yticks(ticks=np.arange(0,1.01,0.1), labels=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1], fontsize=14)
    axis[1].set_title("(b) Four Disorder + Healthy MENTAL EO", fontsize=16, weight="bold") 

    print("\n\nMulti-class prediction EO:")
    #print(f"\tMENTAL e2: {e2_accs[999]}")
    print(f"\tMENTAL e3: {e3_accs[999]}")
    #print(f"\tMENTAL e4: {e4_accs[1999]}")
    #print(f"\tMENTAL e5: {e5_accs[1999]}\n")

    #print(f"\tmax MENTAL e2: {np.max(e2_accs)}")
    print(f"\tmax MENTAL e3: {np.max(e3_accs)}")
    print(np.where(e3_accs == 0.7696969696969697))
    #print(f"\tmax MENTAL e4: {np.max(e4_accs)}")
    #print(f"\tmax MENTAL e5: {np.max(e5_accs)}")

    #figure.legend(loc="lower center", fontsize=20)
    #plt.subplots_adjust(hspace = 0.3, wspace=0.15)
    plt.tight_layout()
    #plt.show()
    plt.savefig("top5_mental")


#plot_MENTAL_MultiClass_Comparison()


def plot_MENTAL_MultiClass_Comparison_top5():

    nc = "tab:blue"
    ic = "tab:orange"
    oc = "tab:green"

    figure, axis = plt.subplots(1, 2, figsize=(15, 5)) 
  
    figure.suptitle("Accuracy of MENTAL (MDD, ADHD, SMC, OCD, Healthy)", fontsize=24, weight="bold")

    # For EC Multi Class
    e15_accs = np.load("C:\\Users\\glgre\\Documents\\ResearchCode\\new2_TOP5_1e5_MENTAL_EC_IMPUTED_ACCS.npy", allow_pickle=True)
    #e25_accs = np.load("C:\\Users\\glgre\\Documents\\ResearchCode\\new_TOP5_2e5_MENTAL_EC_IMPUTED_ACCS.npy", allow_pickle=True)
    #e54_accs = np.load("C:\\Users\\glgre\\Documents\\ResearchCode\\new_TOP5_6e4_MENTAL_EC_IMPUTED_ACCS.npy", allow_pickle=True)
    
    axis[0].plot(e15_accs, label="MENTAL 1e5", c=ic) 
    #axis[0].plot(e25_accs, label="MENTAL 2e5", c=nc)
    #axis[0].plot(e54_accs, label="MENTAL 6e4", c=oc)
    axis[0].set_xlabel("Epoch", fontsize=16)
    axis[0].set_ylabel("Accuracy", fontsize=16)
    axis[0].set_xticks(ticks=np.arange(0,5001,500), labels=np.arange(0,5001,500), fontsize=11)
    axis[0].set_yticks(ticks=np.arange(0,1.01,0.1), labels=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1], fontsize=14)
    axis[0].set_title("(a) Four Disorder + Healthy MENTAL EC", fontsize=16, weight="bold") 

    print("\n\nMulti-class prediction EC:")
    print(f"\tMENTAL 1e5: {e15_accs[4999]}")
    #print(f"\tMENTAL 2e5: {e25_accs[4999]}")
    #print(f"\tMENTAL 6e4: {e54_accs[2999]}\n")

    print(f"\tmax MENTAL 1e5: {np.max(e15_accs)}")
    print(np.where(e15_accs == np.max(e15_accs)))
    #print(f"\tmax MENTAL 2e5: {np.max(e25_accs)}")
    #print(f"\tmax MENTAL 6e4: {np.max(e54_accs)}")

    # For EO Multi Class
    e15_accs = np.load("C:\\Users\\glgre\\Documents\\ResearchCode\\new2_TOP5_1e5_MENTAL_EO_IMPUTED_ACCS.npy", allow_pickle=True)
    #e25_accs = np.load("C:\\Users\\glgre\\Documents\\ResearchCode\\new_TOP5_2e5_MENTAL_EO_IMPUTED_ACCS.npy", allow_pickle=True)
    #e54_accs = np.load("C:\\Users\\glgre\\Documents\\ResearchCode\\new_TOP5_6e4_MENTAL_EO_IMPUTED_ACCS.npy", allow_pickle=True)
    
    axis[1].plot(e15_accs, c=ic) 
    #axis[1].plot(e25_accs, c=nc)
    #axis[1].plot(e54_accs, c=oc)
    axis[1].set_xlabel("Epoch", fontsize=16)
    axis[1].set_ylabel("Accuracy", fontsize=16)
    axis[1].set_xticks(ticks=np.arange(0,5001,500), labels=np.arange(0,5001,500), fontsize=11)
    axis[1].set_yticks(ticks=np.arange(0,1.01,0.1), labels=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1], fontsize=14)
    axis[1].set_title("(b) Four Disorder + Healthy MENTAL EO", fontsize=16, weight="bold") 

    print("\n\nMulti-class prediction EO:")
    print(f"\tMENTAL 1e5: {e15_accs[4999]}")
    #print(f"\tMENTAL 2e5: {e25_accs[4999]}")
    #print(f"\tMENTAL 6e4: {e54_accs[2999]}\n")

    print(f"\tmax MENTAL 1e5: {np.max(e15_accs)}")
    print(np.where(e15_accs == np.max(e15_accs)))
    #print(f"\tmax MENTAL 2e5: {np.max(e25_accs)}")
    #print(f"\tmax MENTAL 6e4: {np.max(e54_accs)}")

    #figure.legend(loc="lower center", fontsize=20)
    #plt.subplots_adjust(hspace = 0.3, wspace=0.15)
    plt.tight_layout()
    #plt.show()
    plt.savefig("new2_top5_mental")

#plot_MENTAL_MultiClass_Comparison_top5()

def plot_MENTAL_MultiClass_Comparison_top3():

    nc = "tab:blue"
    ic = "tab:orange"
    oc = "tab:green"

    figure, axis = plt.subplots(1, 2, figsize=(15, 5)) 
  
    figure.suptitle("Accuracy of MENTAL (MDD, ADHD, SMC)", fontsize=24, weight="bold")

    # For EC Multi Class
    e14_accs = np.load("C:\\Users\\glgre\\Documents\\ResearchCode\\TOP3_1e4_MENTAL_EC_IMPUTED_ACCS.npy", allow_pickle=True)
    e15_accs = np.load("C:\\Users\\glgre\\Documents\\ResearchCode\\TOP3_1e5_MENTAL_EC_IMPUTED_ACCS.npy", allow_pickle=True)
    
    axis[0].plot(e14_accs[:1000], label="MENTAL 1e4", c=ic) 
    #axis[0].plot(e15_accs, label="MENTAL 1e5", c=nc)
    axis[0].set_xlabel("Epoch", fontsize=16)
    axis[0].set_ylabel("Accuracy", fontsize=16)
    axis[0].set_xticks(ticks=np.arange(0,1001,100), labels=np.arange(0,1001,100), fontsize=14)
    axis[0].set_yticks(ticks=np.arange(0,1.01,0.1), labels=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1], fontsize=14)
    axis[0].set_title("(a) Three Disorder MENTAL EC", fontsize=16, weight="bold") 

    print("\n\nMulti-class prediction EC:")
    print(f"\tMENTAL 1e4: {e14_accs[999]}")
    print(f"\tMENTAL 1e5: {e15_accs[999]}\n")

    print(f"\tmax MENTAL 1e4: {np.max(e14_accs)}")
    print(f"\tmax MENTAL 1e5: {np.max(e15_accs)}")

    print(np.where(e14_accs == 0.8))

    # For EO Multi Class
    e14_accs = np.load("C:\\Users\\glgre\\Documents\\ResearchCode\\TOP3_1e4_MENTAL_EO_IMPUTED_ACCS.npy", allow_pickle=True)
    e15_accs = np.load("C:\\Users\\glgre\\Documents\\ResearchCode\\TOP3_1e5_MENTAL_EO_IMPUTED_ACCS.npy", allow_pickle=True)
    
    axis[1].plot(e14_accs[:1000], c=ic) 
    #axis[1].plot(e15_accs, c=nc)
    axis[1].set_xlabel("Epoch", fontsize=16)
    axis[1].set_ylabel("Accuracy", fontsize=16)
    axis[1].set_xticks(ticks=np.arange(0,1001,100), labels=np.arange(0,1001,100), fontsize=14)
    axis[1].set_yticks(ticks=np.arange(0,1.01,0.1), labels=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1], fontsize=14)
    axis[1].set_title("(b) Three Disorder MENTAL EO", fontsize=16, weight="bold") 

    print("\n\nMulti-class prediction EO:")
    print(f"\tMENTAL 1e4: {e14_accs[999]}")
    print(f"\tMENTAL 1e5: {e15_accs[999]}\n")

    print(f"\tmax MENTAL 1e4: {np.max(e14_accs)}")
    print(f"\tmax MENTAL 1e5: {np.max(e15_accs)}")

    print(np.where(e14_accs == 0.84))

    #figure.legend(loc="lower center", fontsize=20)
    plt.tight_layout()
    #plt.show()
    plt.savefig("top3_mental")
    #plt.clf()

#plot_MENTAL_MultiClass_Comparison_top3()

def plot_confusion():

    figure, axis = plt.subplots(1, 2, figsize=(15, 7)) 
  
    figure.suptitle("Confusion Matrix of MENTAL (MDD, ADHD, SMC)", fontsize=24, weight="bold")

    # For EC Multi Class
    confusion_ec = np.load("C:\\Users\\glgre\\Documents\\ResearchCode\\TOP3_1e4_MENTAL_EC_IMPUTED_CONFUSION.npy", allow_pickle=True)
    confusion_eo = np.load("C:\\Users\\glgre\\Documents\\ResearchCode\\TOP3_1e4_MENTAL_EO_IMPUTED_CONFUSION.npy", allow_pickle=True)
    
    g = sns.heatmap(confusion_ec[262], annot=True, cmap='Blues', xticklabels=['MDD', 'ADHD', 'SMC'], yticklabels=['MDD', 'ADHD', 'SMC'], cbar=False, ax=axis[0], annot_kws={'size': 25})
    g.set_xticklabels(g.get_xmajorticklabels(), fontsize = 14)
    g.set_yticklabels(g.get_ymajorticklabels(), fontsize = 14)
    axis[0].set_xlabel(xlabel='Predicted Label', fontsize=15, weight="bold")
    axis[0].set_ylabel(ylabel='Actual Label', fontsize=15, weight="bold")
    axis[0].set_title("(a) Three Disorder MENTAL (EC)", fontsize=16, weight="bold") 

    g = sns.heatmap(confusion_eo[174], annot=True, cmap='Blues', xticklabels=['MDD', 'ADHD', 'SMC'], yticklabels=['MDD', 'ADHD', 'SMC'], cbar=False, ax=axis[1], annot_kws={'size': 25})
    g.set_xticklabels(g.get_xmajorticklabels(), fontsize = 14)
    g.set_yticklabels(g.get_ymajorticklabels(), fontsize = 14)
    axis[1].set_xlabel(xlabel='Predicted Label', fontsize=15, weight="bold")
    axis[1].set_ylabel(ylabel='Actual Label', fontsize=15, weight="bold")
    axis[1].set_title("(b) Three Disorder MENTAL (EO)", fontsize=16, weight="bold") 

    #plt.show()
    plt.savefig("Confusions")

#plot_confusion()

def plot_confusion_top5():

    figure, axis = plt.subplots(1, 2, figsize=(15, 7)) 
  
    figure.suptitle("Confusion Matrix of MENTAL (MDD, ADHD, SMC, OCD, Healthy)", fontsize=24, weight="bold")

    # For EC Multi Class
    confusion_ec = np.load("C:\\Users\\glgre\\Documents\\ResearchCode\\new2_TOP5_1e5_MENTAL_EC_IMPUTED_CONFUSION.npy", allow_pickle=True)
    confusion_eo = np.load("C:\\Users\\glgre\\Documents\\ResearchCode\\new2_TOP5_1e5_MENTAL_EO_IMPUTED_CONFUSION.npy", allow_pickle=True)
    
    g = sns.heatmap(confusion_ec[4350], annot=True, cmap='Blues', xticklabels=['MDD', 'ADHD', 'SMC', 'OCD', 'Healthy'], yticklabels=['MDD', 'ADHD', 'SMC', 'OCD', 'Healthy'], cbar=False, ax=axis[0], annot_kws={'size': 20})
    g.set_xticklabels(g.get_xmajorticklabels(), fontsize = 14)
    g.set_yticklabels(g.get_ymajorticklabels(), fontsize = 14)
    axis[0].set_xlabel(xlabel='Predicted Label', fontsize=15, weight="bold")
    axis[0].set_ylabel(ylabel='Actual Label', fontsize=15, weight="bold")
    axis[0].set_title("(a) Four Disorder + Healthy MENTAL (EC)", fontsize=16, weight="bold") 

    g=sns.heatmap(confusion_eo[4875], annot=True, cmap='Blues', xticklabels=['MDD', 'ADHD', 'SMC', 'OCD', 'Healthy'], yticklabels=['MDD', 'ADHD', 'SMC', 'OCD', 'Healthy'], cbar=False, ax=axis[1], annot_kws={'size': 20})
    g.set_xticklabels(g.get_xmajorticklabels(), fontsize = 14)
    g.set_yticklabels(g.get_ymajorticklabels(), fontsize = 14)
    axis[1].set_xlabel(xlabel='Predicted Label', fontsize=15, weight="bold")
    axis[1].set_ylabel(ylabel='Actual Label', fontsize=15, weight="bold")
    axis[1].set_title("(b) Four Disorder + Healthy MENTAL (EO)", fontsize=16, weight="bold") 

    #plt.show()
    plt.savefig("Confusions_top5_new2")

#plot_confusion_top5()

def calculate_metrics_ec():
    confusion_ec = np.load("C:\\Users\\glgre\\Documents\\ResearchCode\\new2_TOP5_1e5_MENTAL_EC_IMPUTED_CONFUSION.npy", allow_pickle=True)
    confusion_ec = confusion_ec[4350]
    
    #print(confusion_ec)
    precs = []
    recalls = []
    for i in range(5):
        prec = confusion_ec[i,i]/np.sum(confusion_ec[i])
        precs.append(prec)

        recall = confusion_ec[i,i]/np.sum(confusion_ec[:,i])
        recalls.append(recall)
    
    prec_tot = sum(precs)/5
    print(f"\nEC Precision: {prec_tot}")

    recall_tot = sum(recalls)/5
    print(f"EC Recall: {recall_tot}")

    f1_score = 2*(prec_tot*recall_tot)/(prec_tot+recall_tot)  
    print(f"EC F1-Score: {f1_score}")

#calculate_metrics_ec()

def calculate_metrics_eo():
    confusion_eo = np.load("C:\\Users\\glgre\\Documents\\ResearchCode\\new2_TOP5_1e5_MENTAL_EO_IMPUTED_CONFUSION.npy", allow_pickle=True)

    confusion_eo = confusion_eo[4875]

    #print(confusion_eo)
    precs = []
    recalls = []
    for i in range(5):
        prec = confusion_eo[i,i]/np.sum(confusion_eo[i])
        precs.append(prec)

        if(np.sum(confusion_eo[:,i]) == 0):
            recalls.append(0.0)
        else:
            recall = confusion_eo[i,i]/np.sum(confusion_eo[:,i])
            recalls.append(recall)
    
    prec_tot = sum(precs)/5
    print(f"\nEO Precision: {prec_tot}")

    recall_tot = sum(recalls)/5
    print(f"EO Recall: {recall_tot}")

    f1_score = 2*(prec_tot*recall_tot)/(prec_tot+recall_tot)  
    print(f"EO F1-Score: {f1_score}")

#calculate_metrics_eo()

def roc_MENTAL_binary():
    figure, ax = plt.subplots(1, 2, figsize=(12, 5))

    figure.suptitle("ROC Curves of MENTAL (One Disorder vs All Disorders)", fontsize=24, weight="bold")

    # get ec roc mdd
    conds_ec = np.load("C:\\Users\\glgre\\Documents\\ResearchCode\\outs\\MENTAL_EC_IMPUTED_MDD_CONDITIONS.npy")
    preds_ec = np.load("C:\\Users\\glgre\\Documents\\ResearchCode\\outs\\MENTAL_EC_IMPUTED_MDD_PREDICTIONS.npy")

    fpr_ec, tpr_ec, thresholds_ec = roc_curve(conds_ec, preds_ec)
    roc_auc_ec = auc(fpr_ec, tpr_ec)

    # get eo roc mdd
    conds_eo = np.load("C:\\Users\\glgre\\Documents\\ResearchCode\\outs\\MENTAL_EO_IMPUTED_MDD_CONDITIONS.npy")
    preds_eo = np.load("C:\\Users\\glgre\\Documents\\ResearchCode\\outs\\MENTAL_EO_IMPUTED_MDD_PREDICTIONS.npy")

    fpr_eo, tpr_eo, thresholds_eo = roc_curve(conds_eo, preds_eo)
    roc_auc_eo = auc(fpr_eo, tpr_eo)

    # get eo roc mdd
    conds_ec_eo = np.load("C:\\Users\\glgre\\Documents\\ResearchCode\\outs\\MENTAL_EC_EO_IMPUTED_MDD_CONDITIONS.npy")
    preds_ec_eo = np.load("C:\\Users\\glgre\\Documents\\ResearchCode\\outs\\MENTAL_EC_EO_IMPUTED_MDD_PREDICTIONS.npy")

    fpr_ec_eo, tpr_ec_eo, thresholds_ec_eo = roc_curve(conds_ec_eo, preds_ec_eo)
    roc_auc_ec_eo = auc(fpr_ec_eo, tpr_ec_eo)

    # plot mdd roc
    ax[0].plot(fpr_ec, tpr_ec, label='EC (area = %0.2f)' % roc_auc_ec)
    ax[0].plot(fpr_eo, tpr_eo, label='EO (area = %0.2f)' % roc_auc_eo)
    ax[0].plot(fpr_ec_eo, tpr_ec_eo, label='EC+EO (area = %0.2f)' % roc_auc_ec_eo)
    ax[0].plot([0, 1], [0, 1], 'k--', label='No Skill')
    ax[0].set_xlim([0.0, 1.0])
    ax[0].set_ylim([0.0, 1.05])
    ax[0].set_xlabel('False Positive Rate', fontsize=14)
    ax[0].set_ylabel('True Positive Rate', fontsize=14)
    ax[0].set_title('(a) MDD', fontsize=16, weight='bold')
    ax[0].legend(loc="lower right") 

    # get ec roc adhd
    conds_ec = np.load("C:\\Users\\glgre\\Documents\\ResearchCode\\outs\\MENTAL_EC_IMPUTED_ADHD_CONDITIONS.npy")
    preds_ec = np.load("C:\\Users\\glgre\\Documents\\ResearchCode\\outs\\MENTAL_EC_IMPUTED_ADHD_PREDICTIONS.npy")

    fpr_ec, tpr_ec, thresholds_ec = roc_curve(conds_ec, preds_ec)
    roc_auc_ec = auc(fpr_ec, tpr_ec)

    # get eo roc adhd
    conds_eo = np.load("C:\\Users\\glgre\\Documents\\ResearchCode\\outs\\MENTAL_EO_IMPUTED_ADHD_CONDITIONS.npy")
    preds_eo = np.load("C:\\Users\\glgre\\Documents\\ResearchCode\\outs\\MENTAL_EO_IMPUTED_ADHD_PREDICTIONS.npy")

    fpr_eo, tpr_eo, thresholds_eo = roc_curve(conds_eo, preds_eo)
    roc_auc_eo = auc(fpr_eo, tpr_eo)

    # get eo roc adhd
    conds_ec_eo = np.load("C:\\Users\\glgre\\Documents\\ResearchCode\\outs\\MENTAL_EC_EO_IMPUTED_ADHD_CONDITIONS.npy")
    preds_ec_eo = np.load("C:\\Users\\glgre\\Documents\\ResearchCode\\outs\\MENTAL_EC_EO_IMPUTED_ADHD_PREDICTIONS.npy")

    fpr_ec_eo, tpr_ec_eo, thresholds_ec_eo = roc_curve(conds_ec_eo, preds_ec_eo)
    roc_auc_ec_eo = auc(fpr_ec_eo, tpr_ec_eo)

   # plot adhd roc
    ax[1].plot(fpr_ec, tpr_ec, label='EC (area = %0.2f)' % roc_auc_ec)
    ax[1].plot(fpr_eo, tpr_eo, label='EO (area = %0.2f)' % roc_auc_eo)
    ax[1].plot(fpr_ec_eo, tpr_ec_eo, label='EC+EO (area = %0.2f)' % roc_auc_ec_eo)
    ax[1].plot([0, 1], [0, 1], 'k--', label='No Skill')
    ax[1].set_xlim([0.0, 1.0])
    ax[1].set_ylim([0.0, 1.05])
    ax[1].set_xlabel('False Positive Rate', fontsize=14)
    ax[1].set_ylabel('True Positive Rate', fontsize=14)
    ax[1].set_title('(b) ADHD', fontsize=16, weight='bold')
    ax[1].legend(loc="lower right") 

    plt.tight_layout()
    plt.savefig("mental_binary_roc")

#roc_MENTAL_binary()

def roc_MENTAL_top3():
    figure, ax = plt.subplots(1, 2, figsize=(12, 5))

    figure.suptitle("ROC Curves of MENTAL (MDD, ADHD, SMC)", fontsize=24, weight="bold")

    # get ec roc top3
    conds_ec = np.load("C:\\Users\\glgre\\Documents\\ResearchCode\\outs\\MENTAL_EC_TOP3_IMPUTED_CONDITIONS.npy")
    preds_ec = np.load("C:\\Users\\glgre\\Documents\\ResearchCode\\outs\\MENTAL_EC_TOP3_IMPUTED_PREDICTIONS.npy")

    fpr_ec = dict()
    tpr_ec = dict()
    roc_auc_ec = dict()
    for i in range(3):
        fpr_ec[i], tpr_ec[i], _ = roc_curve(conds_ec[:, i], preds_ec[:, i])
        roc_auc_ec[i] = auc(fpr_ec[i], tpr_ec[i])

    # Compute micro-average ROC curve and ROC area
    fpr_ec["micro"], tpr_ec["micro"], _ = roc_curve(conds_ec.ravel(), preds_ec.ravel())
    roc_auc_ec["micro"] = auc(fpr_ec["micro"], tpr_ec["micro"])


    # plot top3 ec roc
    ax[0].plot(fpr_ec["micro"], tpr_ec["micro"], label='EC (area = {0:0.2f})' ''.format(roc_auc_ec["micro"]), color='tab:orange')
    ax[0].plot([0, 1], [0, 1], 'k--', label='No Skill')
    ax[0].set_xlim([0.0, 1.0])
    ax[0].set_ylim([0.0, 1.05])
    ax[0].set_xlabel('False Positive Rate', fontsize=14)
    ax[0].set_ylabel('True Positive Rate', fontsize=14)
    ax[0].set_title('(a) Three Disorder MENTAL (EC)', fontsize=16, weight='bold')
    ax[0].legend(loc="lower right", fontsize=14) 


    # get eo roc top3
    conds_eo = np.load("C:\\Users\\glgre\\Documents\\ResearchCode\\outs\\MENTAL_EO_TOP3_IMPUTED_CONDITIONS.npy")
    preds_eo = np.load("C:\\Users\\glgre\\Documents\\ResearchCode\\outs\\MENTAL_EO_TOP3_IMPUTED_PREDICTIONS.npy")

    fpr_eo = dict()
    tpr_eo = dict()
    roc_auc_eo = dict()
    for i in range(3):
        fpr_eo[i], tpr_eo[i], _ = roc_curve(conds_eo[:, i], preds_eo[:, i])
        roc_auc_eo[i] = auc(fpr_eo[i], tpr_eo[i])

    # Compute micro-average ROC curve and ROC area
    fpr_eo["micro"], tpr_eo["micro"], _ = roc_curve(conds_eo.ravel(), preds_eo.ravel())
    roc_auc_eo["micro"] = auc(fpr_eo["micro"], tpr_eo["micro"])

    # plot top3 eo roc
    ax[1].plot(fpr_eo["micro"], tpr_eo["micro"], label='EO (area = {0:0.2f})' ''.format(roc_auc_eo["micro"]), color='tab:orange')
    ax[1].plot([0, 1], [0, 1], 'k--', label='No Skill')
    ax[1].set_xlim([0.0, 1.0])
    ax[1].set_ylim([0.0, 1.05])
    ax[1].set_xlabel('False Positive Rate', fontsize=14)
    ax[1].set_ylabel('True Positive Rate', fontsize=14)
    ax[1].set_title('(b) Three Disorder MENTAL (EO)', fontsize=16, weight='bold')
    ax[1].legend(loc="lower right", fontsize=14) 

    plt.tight_layout()
    #plt.show()
    plt.savefig("mental_top3_roc")

#roc_MENTAL_top3()

def roc_MENTAL_top5():
    figure, ax = plt.subplots(1, 2, figsize=(12, 5))

    figure.suptitle("ROC Curves of MENTAL (MDD, ADHD, SMC, OCD, Healthy)", fontsize=24, weight="bold")

    # get ec roc top 5
    conds_ec = np.load("C:\\Users\\glgre\\Documents\\ResearchCode\\outs\\MENTAL_EC_TOP5_IMPUTED_CONDITIONS.npy")
    preds_ec = np.load("C:\\Users\\glgre\\Documents\\ResearchCode\\outs\\MENTAL_EC_TOP5_IMPUTED_PREDICTIONS.npy")

    fpr_ec = dict()
    tpr_ec = dict()
    roc_auc_ec = dict()
    for i in range(3):
        fpr_ec[i], tpr_ec[i], _ = roc_curve(conds_ec[:, i], preds_ec[:, i])
        roc_auc_ec[i] = auc(fpr_ec[i], tpr_ec[i])

    # Compute micro-average ROC curve and ROC area
    fpr_ec["micro"], tpr_ec["micro"], _ = roc_curve(conds_ec.ravel(), preds_ec.ravel())
    roc_auc_ec["micro"] = auc(fpr_ec["micro"], tpr_ec["micro"])

    # plot top5 roc
    ax[0].plot(fpr_ec["micro"], tpr_ec["micro"], label='EC (area = {0:0.2f})' ''.format(roc_auc_ec["micro"]), color='tab:orange')
    ax[0].plot([0, 1], [0, 1], 'k--', label='No Skill')
    ax[0].set_xlim([0.0, 1.0])
    ax[0].set_ylim([0.0, 1.05])
    ax[0].set_xlabel('False Positive Rate', fontsize=14)
    ax[0].set_ylabel('True Positive Rate', fontsize=14)
    ax[0].set_title('(a) Four Disorder + Healthy MENTAL (EC)', fontsize=16, weight='bold')
    ax[0].legend(loc="lower right", fontsize=14) 

    # get eo roc top5
    conds_eo = np.load("C:\\Users\\glgre\\Documents\\ResearchCode\\outs\\MENTAL_EO_TOP5_IMPUTED_CONDITIONS.npy")
    preds_eo = np.load("C:\\Users\\glgre\\Documents\\ResearchCode\\outs\\MENTAL_EO_TOP5_IMPUTED_PREDICTIONS.npy")

    fpr_eo = dict()
    tpr_eo = dict()
    roc_auc_eo = dict()
    for i in range(3):
        fpr_eo[i], tpr_eo[i], _ = roc_curve(conds_eo[:, i], preds_eo[:, i])
        roc_auc_eo[i] = auc(fpr_eo[i], tpr_eo[i])

    # Compute micro-average ROC curve and ROC area
    fpr_eo["micro"], tpr_eo["micro"], _ = roc_curve(conds_eo.ravel(), preds_eo.ravel())
    roc_auc_eo["micro"] = auc(fpr_eo["micro"], tpr_eo["micro"])

    # plot top5 roc
    ax[1].plot(fpr_eo["micro"], tpr_eo["micro"], label='EO (area = {0:0.2f})' ''.format(roc_auc_eo["micro"]), color='tab:orange')
    ax[1].plot([0, 1], [0, 1], 'k--', label='No Skill')
    ax[1].set_xlim([0.0, 1.0])
    ax[1].set_ylim([0.0, 1.05])
    ax[1].set_xlabel('False Positive Rate', fontsize=14)
    ax[1].set_ylabel('True Positive Rate', fontsize=14)
    ax[1].set_title('(b) Four Disorder + Healthy MENTAL (EO)', fontsize=16, weight='bold')
    ax[1].legend(loc="lower right", fontsize=14) 

    plt.tight_layout()
    plt.savefig("mental_top5_roc")

#roc_MENTAL_top5()

def roc_MENTAL_under_binary():
    figure, ax = plt.subplots(1, 2, figsize=(12, 5))

    figure.suptitle("ROC Curves of MENTAL (One Disorder vs Healthy)", fontsize=24, weight="bold")

    # get ec roc mdd
    conds_ec = np.load("C:\\Users\\glgre\\Documents\\ResearchCode\\underouts\\MENTAL_EC_HEALTH_MDD_CONDITIONS.npy")
    preds_ec = np.load("C:\\Users\\glgre\\Documents\\ResearchCode\\underouts\\MENTAL_EC_HEALTH_MDD_PREDICTIONS.npy")
    #print(conds_ec)
    #print(preds_ec)

    fpr_ec, tpr_ec, thresholds_ec = roc_curve(conds_ec, preds_ec)
    roc_auc_ec = auc(fpr_ec, tpr_ec)

    # get eo roc mdd
    conds_eo = np.load("C:\\Users\\glgre\\Documents\\ResearchCode\\underouts\\MENTAL_EO_HEALTH_MDD_CONDITIONS.npy")
    preds_eo = np.load("C:\\Users\\glgre\\Documents\\ResearchCode\\underouts\\MENTAL_EO_HEALTH_MDD_PREDICTIONS.npy")
    #print(conds_eo)
    #print(preds_eo)

    fpr_eo, tpr_eo, thresholds_eo = roc_curve(conds_eo, preds_eo)
    roc_auc_eo = auc(fpr_eo, tpr_eo)

    # get eo roc mdd
    conds_ec_eo = np.load("C:\\Users\\glgre\\Documents\\ResearchCode\\underouts\\MENTAL_EC_EO_HEALTH_MDD_CONDITIONS.npy")
    preds_ec_eo = np.load("C:\\Users\\glgre\\Documents\\ResearchCode\\underouts\\MENTAL_EC_EO_HEALTH_MDD_PREDICTIONS.npy")

    fpr_ec_eo, tpr_ec_eo, thresholds_ec_eo = roc_curve(conds_ec_eo, preds_ec_eo)
    roc_auc_ec_eo = auc(fpr_ec_eo, tpr_ec_eo)

    # plot mdd roc
    ax[0].plot(fpr_ec, tpr_ec, label='EC (area = %0.2f)' % roc_auc_ec)
    ax[0].plot(fpr_eo, tpr_eo, label='EO (area = %0.2f)' % roc_auc_eo)
    ax[0].plot(fpr_ec_eo, tpr_ec_eo, label='EC+EO (area = %0.2f)' % roc_auc_ec_eo)
    ax[0].plot([0, 1], [0, 1], 'k--', label='No Skill')
    ax[0].set_xlim([0.0, 1.0])
    ax[0].set_ylim([0.0, 1.05])
    ax[0].set_xlabel('False Positive Rate', fontsize=14)
    ax[0].set_ylabel('True Positive Rate', fontsize=14)
    ax[0].set_title('(a) MDD', fontsize=16, weight='bold')
    ax[0].legend(loc="lower right") 

    # get ec roc adhd
    conds_ec = np.load("C:\\Users\\glgre\\Documents\\ResearchCode\\underouts\\MENTAL_EC_HEALTH_ADHD_CONDITIONS.npy")
    preds_ec = np.load("C:\\Users\\glgre\\Documents\\ResearchCode\\underouts\\MENTAL_EC_HEALTH_ADHD_PREDICTIONS.npy")

    fpr_ec, tpr_ec, thresholds_ec = roc_curve(conds_ec, preds_ec)
    roc_auc_ec = auc(fpr_ec, tpr_ec)

    # get eo roc adhd
    conds_eo = np.load("C:\\Users\\glgre\\Documents\\ResearchCode\\underouts\\MENTAL_EO_HEALTH_ADHD_CONDITIONS.npy")
    preds_eo = np.load("C:\\Users\\glgre\\Documents\\ResearchCode\\underouts\\MENTAL_EO_HEALTH_ADHD_PREDICTIONS.npy")

    fpr_eo, tpr_eo, thresholds_eo = roc_curve(conds_eo, preds_eo)
    roc_auc_eo = auc(fpr_eo, tpr_eo)

    # get eo roc adhd
    conds_ec_eo = np.load("C:\\Users\\glgre\\Documents\\ResearchCode\\underouts\\MENTAL_EC_EO_HEALTH_ADHD_CONDITIONS.npy")
    preds_ec_eo = np.load("C:\\Users\\glgre\\Documents\\ResearchCode\\underouts\\MENTAL_EC_EO_HEALTH_ADHD_PREDICTIONS.npy")

    fpr_ec_eo, tpr_ec_eo, thresholds_ec_eo = roc_curve(conds_ec_eo, preds_ec_eo)
    roc_auc_ec_eo = auc(fpr_ec_eo, tpr_ec_eo)

   # plot adhd roc
    ax[1].plot(fpr_ec, tpr_ec, label='EC (area = %0.2f)' % roc_auc_ec)
    ax[1].plot(fpr_eo, tpr_eo, label='EO (area = %0.2f)' % roc_auc_eo)
    ax[1].plot(fpr_ec_eo, tpr_ec_eo, label='EC+EO (area = %0.2f)' % roc_auc_ec_eo)
    ax[1].plot([0, 1], [0, 1], 'k--', label='No Skill')
    ax[1].set_xlim([0.0, 1.0])
    ax[1].set_ylim([0.0, 1.05])
    ax[1].set_xlabel('False Positive Rate', fontsize=14)
    ax[1].set_ylabel('True Positive Rate', fontsize=14)
    ax[1].set_title('(b) ADHD', fontsize=16, weight='bold')
    ax[1].legend(loc="lower right") 

    plt.tight_layout()
    plt.show()

roc_MENTAL_under_binary()