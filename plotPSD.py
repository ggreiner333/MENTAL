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
  
    figure.suptitle("Accuracy of MENTAL vs. EEG Only", fontsize=24, weight="bold")

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
  
    figure.suptitle("Accuracy of MENTAL vs. EEG Only (Healthy vs Non)", fontsize=24, weight="bold")

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

    figure.legend(loc="lower center", fontsize=20)
    plt.subplots_adjust(hspace = 0.3, wspace=0.15)
    #plt.tight_layout()
    plt.savefig("mental_eeg_control_comparison")
    
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

    figure, axis = plt.subplots(2, 2, figsize=(15, 14)) 
  
    figure.suptitle("Accuracy of MENTAL vs. EEG Only", fontsize=24, weight="bold")

    # For ADHD EC
    adhd_accs_normal = np.load("C:\\Users\\glgre\\Documents\\ResearchCode\\EC_ONLY_ADHD_ACCS.npy", allow_pickle=True)
    #adhd_accs_mental = np.load("C:\\Users\\glgre\\Documents\\ResearchCode\\ADHD_MENTAL_EC_ACCS.npy", allow_pickle=True)
    adhd_accs_impute = np.load("C:\\Users\\glgre\\Documents\\ResearchCode\\MENTAL_EC_IMPUTED_ADHD_ACCS.npy", allow_pickle=True)
    axis[0, 0].plot(adhd_accs_normal, label="EEG Only", c=nc)
    #axis[0, 0].plot(adhd_accs_mental, label="MENTAL"  , c=mc)
    axis[0, 0].plot(adhd_accs_impute, label="Imputed" , c=ic) 
    axis[0, 0].set_xlabel("Epoch", fontsize=16)
    axis[0, 0].set_ylabel("Accuracy", fontsize=16)
    axis[0, 0].set_xticks(ticks=np.arange(0,1001,100), labels=np.arange(0,1001,100), fontsize=14)
    axis[0, 0].set_yticks(ticks=np.arange(0,1.01,0.1), labels=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1], fontsize=14)
    axis[0, 0].set_title("(a) ADHD (EC)", fontsize=16, weight="bold") 
    
    # For MDD EC
    mdd_accs_normal = np.load('EC_ONLY_ACCS.npy', allow_pickle=True)
    #mdd_accs_mental = np.load('diff_MENTAL_EC_ACCS.npy', allow_pickle=True)
    mdd_accs_impute = np.load("C:\\Users\\glgre\\Documents\\ResearchCode\\MENTAL_EC_IMPUTED_MDD_ACCS.npy", allow_pickle=True)
    axis[0, 1].plot(mdd_accs_normal, c=nc)
    #axis[0, 1].plot(mdd_accs_mental, c=mc) 
    axis[0, 1].plot(mdd_accs_impute, c=ic) 
    axis[0, 1].set_xlabel("Epoch", fontsize=16)
    axis[0, 1].set_ylabel("Accuracy", fontsize=16)
    axis[0, 1].set_xticks(ticks=np.arange(0,1001,100), labels=np.arange(0,1001,100), fontsize=14)
    axis[0, 1].set_yticks(ticks=np.arange(0,1.01,0.1), labels=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1], fontsize=14)
    axis[0, 1].set_title("(b) MDD (EC)", fontsize=16, weight="bold") 

    # For ADHD EO
    adhd_accs_normal = np.load("C:\\Users\\glgre\\Documents\\ResearchCode\\EO_ONLY_ADHD_ACCS.npy", allow_pickle=True)
    #adhd_accs_mental = np.load("C:\\Users\\glgre\\Documents\\ResearchCode\\ADHD_MENTAL_EO_ACCS.npy", allow_pickle=True)
    adhd_accs_impute = np.load("C:\\Users\\glgre\\Documents\\ResearchCode\\MENTAL_EO_IMPUTED_ADHD_ACCS.npy", allow_pickle=True)
    axis[1, 0].plot(adhd_accs_normal, c= nc)
    #axis[1, 0].plot(adhd_accs_mental, c= mc)
    axis[1, 0].plot(adhd_accs_impute, c=ic) 
    axis[1, 0].set_xlabel("Epoch", fontsize=16)
    axis[1, 0].set_ylabel("Accuracy", fontsize=16)
    axis[1, 0].set_xticks(ticks=np.arange(0,1001,100), labels=np.arange(0,1001,100), fontsize=14)
    axis[1, 0].set_yticks(ticks=np.arange(0,1.01,0.1), labels=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1], fontsize=14)
    axis[1, 0].set_title("(c) ADHD (EO)", fontsize=16, weight="bold") 
    
    # For MDD EO
    mdd_accs_normal = np.load('EO_ONLY_ACCS.npy', allow_pickle=True)
    #mdd_accs_mental = np.load('diff_MENTAL_EO_ACCS.npy', allow_pickle=True)
    mdd_accs_impute = np.load("C:\\Users\\glgre\\Documents\\ResearchCode\\MENTAL_EO_IMPUTED_MDD_ACCS.npy", allow_pickle=True)
    axis[1, 1].plot(mdd_accs_normal, c= nc)
    #axis[1, 1].plot(mdd_accs_mental, c= mc) 
    axis[1, 1].plot(mdd_accs_impute, c= ic) 
    axis[1, 1].set_xlabel("Epoch", fontsize=16)
    axis[1, 1].set_ylabel("Accuracy", fontsize=16)
    axis[1, 1].set_xticks(ticks=np.arange(0,1001,100), labels=np.arange(0,1001,100), fontsize=14)
    axis[1, 1].set_yticks(ticks=np.arange(0,1.01,0.1), labels=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1], fontsize=14)
    axis[1, 1].set_title("(d) MDD (EO)", fontsize=16, weight="bold") 

    figure.legend(loc="lower center", fontsize=10)
    plt.subplots_adjust(hspace = 0.3, wspace=0.15)
    plt.show()
    #plt.savefig("imputedcomparison")
    
plot_MENTAL_EEG_Imputed_Comparison()