import os
from os.path import dirname
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

import mne


##################################################################################################
##################################################################################################
##################################################################################################

# path of preprocessed EEG data
preprocess_file_path = 'TDBRAIN/preprocessed'

# path of directory where we will save the PSD features
psds_path = 'TDBRAIN/PSD_all'

# channels
all_included = [ "Fp1",  "Fp2",   "F7",   "F3",    "Fz",    "F4",   "F8", "FC3", "FCz", "FC4", "T7", "C3", "Cz",
                  "C4",   "T8",  "CP3",  "CPz",   "CP4",    "P7",   "P3",  "Pz",  "P4",  "P8", "O1", "Oz", "O2" ]

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

#test()

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

#plot_box()

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

#get_ind()

def plot_test():
    accs = np.load('diff_MENTAL_EO_ACCS_epoch_200.npy', allow_pickle=True)

    labels = np.arange(0, 201, 1)
    plt.figure(figsize=(15,10))
    plt.plot(labels, accs)
    plt.title("Accuracy of diff EC MENTAL for 200 epochs, batch size 15")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.yticks(ticks=np.arange(0,1.01,0.1))

    plt.savefig("diff_mental_epoch1000_b15_w6_l3_accuracy_ec")
    plt.clf()

plot_test()