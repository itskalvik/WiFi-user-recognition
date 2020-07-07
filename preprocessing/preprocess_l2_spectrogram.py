#!/usr/bin/env python3

from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from skimage import transform
from scipy.io import loadmat
from scipy import signal
import numpy as np
import argparse
import pickle
import time
import h5py
import math
import sys
import os


def get_csi(x):
    x = np.squeeze(x["csi_trace"])
    data = []
    for i in range(x.shape[0]):
        array = {}
        array["timestamp_low"] = np.squeeze(x[i][0][0][0])
        array["bfee_count"] = np.squeeze(x[i][0][0][1])
        array["Nrx"] = np.squeeze(x[i][0][0][2])
        array["Ntx"] = np.squeeze(x[i][0][0][3])
        array["rssi_a"] = np.squeeze(x[i][0][0][4])
        array["rssi_b"] = np.squeeze(x[i][0][0][5])
        array["rssi_c"] = np.squeeze(x[i][0][0][6])
        array["noise"] = np.squeeze(x[i][0][0][7])
        array["agc"] = np.squeeze(x[i][0][0][8])
        array["perm"] = np.squeeze(x[i][0][0][9])
        array["rate"] = np.squeeze(x[i][0][0][10])
        array["csi"] = np.squeeze(x[i][0][0][11])

        data.append(array)
    return data


def phase_correction(ph_raw):
    m = np.arange(-28, 29)
    Tp = np.unwrap(ph_raw)
    k_param = (Tp[29] - Tp[0]) / (m[29] - m[0])
    b_param = np.sum(Tp) * (1 / 30)

    correct_phase = []
    for i in range(30):
        correct_phase.append(Tp[i] - k_param * m[i] - b_param)
    return correct_phase


# 3 x 3 MIMO Matrix format
# [h11 h12 h13
# h21 h22 h23
# h31 h32 h33]
def apply_phcorrect(ph_raw):
    mimo_mat = np.rollaxis(ph_raw, 2, 0)
    mimo_mat = np.reshape(mimo_mat, (30, 9))

    crct_ph = []
    for col in range(9):
        crct_ph.append(phase_correction(np.array(mimo_mat)[:, col]))

    stack_crc_ph = np.vstack(crct_ph).T

    restore_ph_mat = []
    for i in range(30):
        restore_ph_mat.append(stack_crc_ph[i, :].reshape((3, 3)))
    return np.array(restore_ph_mat).T


def power_delay_profile(data, keep_bins=10):
    data = np.concatenate([
        np.zeros_like(data[..., 0:1]), data,
        np.expand_dims(data[..., -1], axis=-1)
    ],
                          axis=-1)
    pdf = np.fft.irfft(data, axis=-1)
    pdf[..., keep_bins:] = 0
    return np.fft.fft(pdf, n=(data.shape[-1] * 2) + 2,
                      axis=-1)[..., 1:data.shape[-1] - 1]


def fill_gaps(csi_trace, technique):
    amp_data = []
    ph_data = []

    for ind in range(len(csi_trace)):
        csi_entry = csi_trace[ind]

        scaled_csi = power_delay_profile(get_scaled_csi(csi_entry))
        amp = np.absolute(scaled_csi)
        ph = np.angle(scaled_csi)

        amp_temp = []
        ph_temp = []

        if technique == 'fill':
            if csi_trace[ind]['Ntx'] == 1:
                ph = np.expand_dims(ph, axis=0)
                amp = np.expand_dims(amp, axis=0)
                for i in range(30):
                    amp_temp.append(
                        np.append(amp[:, :, i],
                                  np.zeros((2, 3)) + np.nan).reshape((3, 3)))
                    ph_temp.append(
                        np.append(ph[:, :, i],
                                  np.zeros((2, 3)) + np.nan).reshape((3, 3)))
                amp_data.append(np.array(amp_temp).flatten())
                ph_data.append(apply_phcorrect(ph_temp).flatten())

            elif csi_trace[ind]['Ntx'] == 2:
                for i in range(30):
                    amp_temp.append(
                        np.append(amp[:, :, i],
                                  np.zeros((1, 3)) + np.nan).reshape((3, 3)))
                    ph_temp.append(
                        np.append(ph[:, :, i],
                                  np.zeros((1, 3)) + np.nan).reshape((3, 3)))
                amp_data.append(np.array(amp_temp).flatten())
                ph_data.append(apply_phcorrect(ph_temp).flatten())

            elif csi_trace[ind]['Ntx'] == 3:
                amp_data.append(np.array(amp).T.flatten())
                ph_data.append(apply_phcorrect(ph).T.flatten())

        elif technique == 'mean':
            if csi_trace[ind]['Ntx'] == 1:
                ph = np.expand_dims(ph, axis=0)
                amp = np.expand_dims(amp, axis=0)

                mean_amp = np.mean(amp)
                mean_ph = np.mean(ph)

                for i in range(30):
                    amp_temp.append(
                        np.append(amp[:, :, i],
                                  np.zeros((2, 3)) + mean_amp).reshape((3, 3)))
                    ph_temp.append(
                        np.append(ph[:, :, i],
                                  np.zeros((2, 3)) + mean_ph).reshape((3, 3)))
                ph_temp = np.array(ph_temp).T
                amp_data.append(np.array(amp_temp).flatten())
                ph_data.append(apply_phcorrect(ph_temp).flatten())

            elif csi_trace[ind]['Ntx'] == 2:
                mean_amp = np.mean(amp)
                mean_ph = np.mean(ph)
                for i in range(30):
                    amp_temp.append(
                        np.append(amp[:, :, i],
                                  np.zeros((1, 3)) + mean_amp).reshape((3, 3)))
                    ph_temp.append(
                        np.append(ph[:, :, i],
                                  np.zeros((1, 3)) + mean_ph).reshape((3, 3)))
                ph_temp = np.array(ph_temp).T
                amp_data.append(np.array(amp_temp).flatten())
                ph_data.append(apply_phcorrect(ph_temp).flatten())

            elif csi_trace[ind]['Ntx'] == 3:
                amp_data.append(np.array(amp).T.flatten())
                ph_data.append(apply_phcorrect(ph).T.flatten())

    return np.hstack([amp_data, ph_data])


def dbinv(x):
    return np.power(10, (np.array(x) / 10))


def get_total_rss(csi_st):
    rssi_mag = 0
    if csi_st['rssi_a'] != 0:
        rssi_mag = rssi_mag + dbinv(csi_st['rssi_a'])

    if csi_st['rssi_b'] != 0:
        rssi_mag = rssi_mag + dbinv(csi_st['rssi_b'])

    if csi_st['rssi_c'] != 0:
        rssi_mag = rssi_mag + dbinv(csi_st['rssi_c'])

    return 10 * np.log10(rssi_mag) - 44 - csi_st['agc']


def get_scaled_csi(csi_st):
    csi = csi_st['csi']

    csi_sq = np.multiply(csi, np.conj(csi))
    csi_pwr = np.sum(csi_sq[:])
    rssi_pwr = dbinv(get_total_rss(csi_st))

    scale = rssi_pwr / (csi_pwr / 30)

    if (csi_st['noise'] == -127):
        noise_db = -92
    else:
        noise_db = csi_st['noise']

    thermal_noise_pwr = dbinv(noise_db)
    quant_error_pwr = scale * (csi_st['Nrx'] * csi_st['Ntx'])
    total_noise_pwr = thermal_noise_pwr + quant_error_pwr

    ret = csi * np.sqrt(scale / total_noise_pwr)
    if csi_st['Ntx'] == 2:
        ret = ret * np.sqrt(2)
    elif csi_st['Ntx'] == 3:
        ret = ret * np.sqrt(dbinv(4.5))

    return ret


def read_samples(dataset_path, endswith=".csv"):
    datapaths, labels = list(), list()
    label = 0
    classes = sorted([
        f for f in os.listdir(dataset_path) if
        os.path.isdir(os.path.join(dataset_path, f)) and not f.startswith('.')
    ])

    # List each sub-directory (the classes)
    for c in classes:
        c_dir = os.path.join(dataset_path, c)
        walk = os.walk(c_dir).__next__()
        # Add each image to the training set
        for sample in walk[2]:
            # Only keeps csv samples
            if sample.endswith(endswith):
                datapaths.append(os.path.join(c_dir, sample))
                labels.append(label)
        label += 1

    return datapaths, labels, classes


def smooth(x, window_len):
    s = np.r_[x[window_len - 1:0:-1], x, x[-2:-window_len - 1:-1]]
    w = np.hanning(window_len)
    y = np.convolve(w / w.sum(), s, mode='valid')
    return y[(window_len // 2):-(window_len // 2)]


def spectrogram(data,
                nfft=2048,
                window_size=1024,
                fs=2000,
                fc=10,
                n_components=30):
    freq_res = (fs / 2) / (nfft / 2)
    target_fft = int(np.ceil(150 / freq_res))

    cmap_obj = plt.cm.ScalarMappable(cmap="jet")
    cmap_obj.set_clim(-16, 6)

    pca = PCA(n_components=n_components)
    data = pca.fit_transform(data)

    b, a = signal.butter(8, fc / (fs / 2), 'high')
    data = signal.lfilter(b, a, data, axis=0)

    for i in range(n_components):
        f, t, s = signal.spectrogram(data[:, i],
                                     fs=fs,
                                     window=signal.windows.gaussian(
                                         window_size, (window_size - 1) / 5),
                                     noverlap=window_size - 1,
                                     nfft=nfft)
        s_target = np.abs(s)[:target_fft]
        s_target /= np.sum(s_target, axis=0)
        s_target -= np.mean(s_target)
        s_target[np.where(s_target < 0)] = 0
        if (i == 0):
            s_new = s_target
        else:
            s_new += s_target

    s_new = 20 * np.log10(s_new + 1e-9)
    s_new = transform.resize(s_new, (512, 512))
    s_new = np.flip(s_new, axis=0)
    return cmap_obj.to_rgba(s_new)[:, :, :3]


def compute_data(file_path, sampling, cols1, cols2, label):
    if (not os.path.isfile(file_path)):
        raise ValueError("File dosn't exits")

    csi_trace = get_csi(loadmat(file_path))[:10000]
    csi_trace = csi_trace[::sampling]
    csi_trace = fill_gaps(csi_trace, technique='mean')[:, cols1:cols2]

    return spectrogram(csi_trace), label


#******************************************************************************#

#sampling 1: 8000
#sampling 2: 4000
#sampling 4: 2000
#sampling 8: 1000
#sampling 16: 500

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True, help="source data directory")
ap.add_argument(
    "--dataset",
    required=True,
    help=
    "destination h5py file for dataset, can be (False) to disable saving dataset"
)
ap.add_argument("--sampling",
                required=True,
                type=int,
                help="sampling rate for data")
ap.add_argument("--cols",
                required=True,
                help="Data that needs to be computed AMP/PH/ALL")
ap.add_argument(
    "--mc",
    required=True,
    type=int,
    help=
    "set value to (1) if generating a single dataset else give dataset index (greater than 1) [If generating single dataset train_test_split function will get seed of 42, if generating multiple sets seed will not be set, for random splits]"
)

args = vars(ap.parse_args())

src_path = args["src"].strip()
dataset_file = args["dataset"].strip()
sampling = args["sampling"]
cols = args["cols"].strip()
mc = args["mc"]

if mc == 1:
    seed = 42
else:
    seed = None

if cols == "AMP":
    cols1 = 0
    cols2 = 270
    cols = 270
elif cols == "PH":
    cols1 = 270
    cols2 = 540
    cols = 270
elif cols == "ALL":
    cols1 = 0
    cols2 = 540
    cols = 540
else:
    raise ValueError(
        "Check cols argument!! Got: {} | Acceptable arguments (AMP/PH/ALL)".
        format(cols))

rows = int(8000 / sampling)

print("rows:", rows, "| cols:", cols1, "-", cols2)
sys.stdout.flush()

files, labels, classes = read_samples(src_path, ".mat")
classes = [n.encode("ascii", "ignore") for n in classes]

dset_X, dset_y = zip(*Parallel(n_jobs=-2)(
    delayed(compute_data)(files[ind], sampling, cols1, cols2, labels[ind])
    for ind in range(len(files))))
dset_X = np.array(dset_X)
dset_y = np.array(dset_y)

for iter in range(mc):
    train_X, test_X, train_y, test_y = train_test_split(dset_X,
                                                        dset_y,
                                                        test_size=0.15,
                                                        random_state=seed,
                                                        stratify=dset_y)

    print("X_train: {} | X_test: {} | y_train: {} | y_test: {}".format(
        train_X.shape, test_X.shape, train_y.shape, test_y.shape))

    if dataset_file != "False":
        if not os.path.exists(os.path.dirname(dataset_file)):
            try:
                os.makedirs(os.path.dirname(dataset_file))
            except OSError as exc:
                if exc.errno != errno.EEXIST:
                    raise

        if mc != 1:
            hf = h5py.File(dataset_file.format(iter), 'w')
        else:
            hf = h5py.File(dataset_file, 'w')

    hf.create_dataset('X_train', data=train_X)
    hf.create_dataset('y_train', data=train_y)
    hf.create_dataset('X_test', data=test_X)
    hf.create_dataset('y_test', data=test_y)
    hf.create_dataset('labels', data=classes)
    hf.create_dataset('sampling', data=sampling)
    hf.create_dataset('cols', data=args["cols"].strip())
    hf.close()

print("finished!!")
