import os
from os.path import dirname
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

import mne


##################################################################################################
##################################################################################################
##################################################################################################


# Frequency Bands

# delta (0–4 Hz)
delta_min = 0
delta_max = 4

# theta (4–8 Hz)
theta_min = 4
theta_max = 8

# alpha (8–12 Hz)
alpha_min = 8
alpha_max = 12

# beta (16–32 Hz)
beta_min = 16
beta_max = 32

# gamma (32–64 Hz)
gamma_min = 32
gamma_max = 64


# Sampling Frequency (500Hz)
sampling_freq = 500 


# Channel information

all_channels = [ "Fp1",  "Fp2",   "F7",   "F3",    "Fz",    "F4",   "F8", "FC3", "FCz", "FC4", "T7", "C3", "Cz",
                  "C4",   "T8",  "CP3",  "CPz",   "CP4",    "P7",   "P3",  "Pz",  "P4",  "P8", "O1", "Oz", "O2", 
           "artifacts", "VEOG", "HEOG", "Erbs", "OrbOcc", "Mass" ]

channel_type = [ "eeg",  "eeg",  "eeg",  "eeg",  "eeg",  "eeg",  "eeg", "eeg", "eeg", "eeg", "eeg", "eeg", "eeg",
                 "eeg",  "eeg",  "eeg",  "eeg",  "eeg",  "eeg",  "eeg", "eeg", "eeg", "eeg", "eeg", "eeg", "eeg", 
                "misc", "misc", "misc", "misc", "misc", "misc" ]

exclude_these = ['artifacts', 'VEOG', 'HEOG', 'Erbs', 'OrbOcc', 'Mass']


# path of preprocessed EEG data
preprocess_file_path = '/data/zhanglab/ggreiner/MENTAL/TDBRAIN/preprocessed'

# path of directory where we will save the PSD features
psds_path = '/data/zhanglab/ggreiner/MENTAL/TDBRAIN/PSD'


##################################################################################################
##################################################################################################
##################################################################################################


def get_psd(file):

    """

    This function extracts PSD values from each channel in the five frequency bands 
    (delta, theta, alpha, beta, gamma) from the 2-min EEG data.

    Parameters
    ----------
        file: location of numpy file that contains the EEG data

    Returns
    ----------
        all_psds: numpy array that contains the psd values

    """


    # Load preprocessed EEG data saved in an .npy file
    loaded = np.load(file, allow_pickle=True)
    data = np.squeeze(loaded['data'])
    channel_names = loaded['labels'].tolist()
    
    # Create an instance of a raw object that we can use to extract PSD from
    info = mne.create_info(ch_names= channel_names, ch_types=channel_type, sfreq= sampling_freq, verbose="error")
    raw = mne.io.RawArray(data, info, verbose="error")

    # Output
    all_psds = []

    for t in range(0,120,2):
        interval_psd =[]
        for name in channel_names:
            if(not exclude_these.__contains__(name)):
                channel_psds = []

                # extract average delta PSD for the given 2-sec interval
                d_psd = raw.compute_psd(fmin=delta_min, fmax=delta_max, tmin=t, tmax=t+2, picks=name, verbose="error")
                d_vals, d_freqs = d_psd.get_data(return_freqs=True)
                avg = np.average(d_vals)
                channel_psds.append(avg)

                # extract average theta PSD for the given 2-sec interval
                t_psd = raw.compute_psd(fmin=theta_min, fmax=theta_max, tmin=t, tmax=t+2, picks=name, verbose="error")
                t_vals, t_freqs = t_psd.get_data(return_freqs=True)
                avg = np.average(t_vals)
                channel_psds.append(avg)

                # extract average alpha PSD for the given 2-sec interval
                a_psd = raw.compute_psd(fmin=alpha_min, fmax=alpha_max, tmin=t, tmax=t+2, picks=name, verbose="error")
                a_vals, a_freqs = a_psd.get_data(return_freqs=True)
                avg = np.average(a_vals)
                channel_psds.append(avg)

                # extract average beta PSD for the given 2-sec interval
                b_psd = raw.compute_psd(fmin=beta_min, fmax=beta_max, tmin=t, tmax=t+2, picks=name, verbose="error")
                b_vals, b_freqs = b_psd.get_data(return_freqs=True)
                avg = np.average(b_vals)
                channel_psds.append(avg)

                # extract average gamma PSD for the given 2-sec interval
                g_psd = raw.compute_psd(fmin=gamma_min, fmax=gamma_max, tmin=t, tmax=t+2, picks=name, verbose="error")
                g_vals, g_freqs = g_psd.get_data(return_freqs=True)
                avg = np.average(g_vals)
                channel_psds.append(avg)


                # add the PSD values for this channel to the interval data
                interval_psd.append(channel_psds)

        # add the PSD values for this interval to total data
        all_psds.append(interval_psd)

    # return a numpy array with the psd information
    final = np.array(all_psds)
    return final


def extract_psds(path, out):

    """

    This function extracts PSD features using get_psd for every individual
    in the specified directory. It then saves the numpy arrays as .npy files
    in the specified output directory.

    Parameters
    ----------
        path: path of the directory that contains the individuals' data
              (Note: this function is designed to accommodate for the 
               directory structure that results from running the 
               preprocessing code that TDBRAIN provided)

        out : path of the directory where the psd information will be saved

    """

    # Gather list of individuals
    individuals = os.listdir(path)
    sn = 0

    for ind in individuals:
        done = False
        # Generate output folders if they don't already exist
        output = os.path.join(out, ind)
        if(not os.path.isdir(output)):
            #done = False
            os.mkdir(output)
        
        if(not done):
            # Generate PSD values for each session the individual has
            sessions_dir = os.path.join(preprocess_file_path, ind)
            for sess in os.listdir(sessions_dir):
                data  = os.path.join(sessions_dir, sess, "eeg")
                files = os.listdir(data)
                sn = sess #saving session info

                # filter for the files that contain the preprocessed EEG recordings
                for f in files:
                    pth = os.path.join(data,f)
                    if os.path.isfile(pth):
                        time =    f.split("_")[-1]
                        sec  = time.split(".")[0]
                        if(int(sec) >= 118):
                            # extract psd values using get_psd
                            res = get_psd(pth)
                            # where to save the file
                            write_to = os.path.join(output, sn + ("_EO" if f.__contains__("EO") else "_EC"))
                            np.save(write_to, res, allow_pickle=True)

extract_psds(preprocess_file_path, psds_path)