from os.path import dirname
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

import mne

#myRaw = mne.io.read_raw_brainvision("sub-87965393_ses-1_task-restEC_eeg.vhdr", preload=True)

#print(myRaw)
#print(myRaw.info)

#mne.viz.plot_raw(myRaw, duration =0.01, start=0.0, show = True)

#myPlot = myRaw.plot(duration = 0.25, start = 0.0, show=True)

#myPlot.save()
#myPlot = myRaw.plot(None, 1.0, 0.0)

#myPlot.show()

all_channels = [  "Fp1",  "Fp2",   "F7",   "F3",   "Fz",     "F4",   "F8", "FC3", "FCz", "FC4", "T7", "C3", "Cz",
                   "C4",   "T8",  "CP3",  "CPz",  "CP4",     "P7",   "P3",  "Pz",  "P4",  "P8", "O1", "Oz", "O2", 
                 "VPVA", "VNVB", "HPHL", "HNHR", "Erbs", "OrbOcc", "Mass"]

def show1():
    raw = mne.io.read_raw_brainvision("sub-87965393_ses-1_task-restEC_eeg.vhdr", preload=True)
    sampling_freq = raw.info["sfreq"]
    start_stop_seconds = np.array([0, 10])
    start_sample, stop_sample = (start_stop_seconds * sampling_freq).astype(int)
    channel_index = 0
    raw_selection = raw[channel_index, start_sample:stop_sample]
    print(raw_selection)
    print(raw.ch_names[0])

    x = raw_selection[1]
    y = raw_selection[0].T
    plt.plot(x, y)
    plt.show()
    
def show2():
    raw = mne.io.read_raw_brainvision("sub-87965393_ses-1_task-restEC_eeg.vhdr", preload=True)
    sampling_freq = raw.info["sfreq"]
    start_stop_seconds = np.array([30, 40])
    start_sample, stop_sample = (start_stop_seconds * sampling_freq).astype(int)
    #channels = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "FC3", "FCz", "FC4"]
    #channels = ["FC3", "FCz", "FC4"]
    channels = ["Pz"]

    raw_selection = raw[channels, start_sample:stop_sample]
    print(raw_selection)
    print(raw.ch_names[0])

    x = raw_selection[1]
    y = raw_selection[0].T
    lines = plt.plot(x, y)
    plt.legend(lines, channels)
    plt.show()
    

def showChannel(chnl, intStart, intEnd, open):
    if(open):
        raw = mne.io.read_raw_brainvision("sub-87965393_ses-1_task-restEO_eeg.vhdr", preload=True)
    else:
        raw = mne.io.read_raw_brainvision("sub-87965393_ses-1_task-restEC_eeg.vhdr", preload=True)
    sampling_freq = raw.info["sfreq"]
    start_stop_seconds = np.array([intStart, intEnd])
    start_sample, stop_sample = (start_stop_seconds * sampling_freq).astype(int)
    channels = [chnl]

    raw_selection = raw[channels, start_sample:stop_sample]
    x = raw_selection[1]
    y = raw_selection[0].T

    lines = plt.plot(x, y)
    plt.legend(lines, channels)
    plt.xlabel("Time")
    plt.ylabel("uV")
    plt.savefig(chnl + "_" + str(intStart) + "_" + str(intEnd) + ".pdf", format="pdf", bbox_inches="tight")
    plt.show()

#for chnl in all_channels:
#    showChannel(chnl, 4, 5, True)

for chnl in all_channels:
    showChannel(chnl, 30, 40, True)



#for chnl in all_channels:
#    showChannel(chnl, 0, 120, True)

#show1()
#show2()

