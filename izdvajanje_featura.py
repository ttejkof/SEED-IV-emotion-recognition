import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import mne
import pprint
from tqdm import tqdm
import pickle
import argparse

args_parser = argparse.ArgumentParser()
args_parser.add_argument("--human_id", type=int, default=1)
args = args_parser.parse_args()
wanted_human_id = args.human_id

# %%

from utils.data import iterate_all, get_feature_shape, iterate_human
from utils.data import Dataset
dataset: Dataset = Dataset.get_dataset(reload=False)

# %%
human_params = pickle.load(open('human_params.pkl', 'rb'))

# %%
montage = mne.channels.read_custom_montage(r'SEED-IV/channel_62_pos.locs')

# %%
def convert_np2mne(data: np.ndarray):
    n_channels = 62
    sampling_freq = 200  # in Hertz
    info = mne.create_info(n_channels, sfreq=sampling_freq)
    ch_names = ["Fp1", "Fpz", "Fp2", "AF3", "AF4", "F7", "F5", "F3", "F1", "Fz", "F2", "F4", "F6", "F8", "FT7", "FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6", "FT8", "T7", "C5", "C3", "C1", "Cz", "C2", "C4", "C6", "T8", "TP7", "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6", "TP8", "P7", "P5", "P3", "P1", "Pz", "P2", "P4", "P6", "P8", "PO7", "PO5", "PO3", "POz", "PO4", "PO6", "PO8", "CB1", "O1", "Oz", "O2", "CB2"]
    ch_types = ["eeg"]*62
    info = mne.create_info(ch_names, ch_types=ch_types, sfreq=sampling_freq)
    eeg = mne.io.RawArray(data/1e6, info)
    eeg.set_montage(montage)
    return eeg

# %%
def eeg_filter(mne_data, low_freq, high_freq):
    return mne_data.copy().filter(low_freq, high_freq, 'eeg')

# %%
def split_abgd(eeg_data):
    eeg_alfa = eeg_data.copy().filter(8, 15, 'eeg')
    eeg_beta = eeg_data.copy().filter(16, 31, 'eeg')
    eeg_gamma = eeg_data.copy().filter(32, 45, 'eeg')
    eeg_teta = eeg_data.copy().filter(4, 7, 'eeg')
    return eeg_alfa, eeg_beta, eeg_gamma, eeg_teta


# %%
import scipy.signal

Fs = 200
def peak2peak(channel):
    poz_pik, mag_pik = mne.preprocessing.peak_finder(channel) #90
    Ts = 1/Fs
    prvo = Ts*poz_pik[1:]
    drugo = Ts*poz_pik[0:-1]
    razlika = prvo - drugo
    PTP = np.mean(razlika)
    return PTP

# Ovo treba proveriti
def calculate_c0(x):
    X = np.fft.fft(x, axis=-1)
    M = np.mean(np.abs(X)**2, axis=-1)
    Y = np.where(X > M, X, 0)
    y = np.fft.ifft(Y, axis=-1)
    A1 = np.sum((x - y)**2, axis=-1)
    A0 = np.sum(x**2, axis=-1)
    return A1 / A0

# %%
# min_time = min(data.shape[1] for data, label, ids in iterate_all(dataset)) / Fs
# min_time = min_time // 5 * 5 # da se zaokruzi na lepe brojeve

# %%
min_time = 40 # OVo treba proveriti

# %%
import mne_features
import antropy
import EntropyHub
import time

# %%
# Treba razmisliiti da li ima smisla da za svaki band pravimo posebne feature
# pod band_id mislim na alpha, beta, gamma, delta
# (session_id, human_id, video_id, epoch_id, band_id, channel_id, feature_id)
# (3, 15, 24, 8, 4, 62, 10)
# Obratiti paznju da je session_id, human_id, video_id mislim da krecu od 1 a ne od 0 ali treba proveriti
# Takodje za pocetak mozda samo izdvojiti 

start_humain_id = wanted_human_id
for wanted_human_id in range(start_humain_id, 16, 5):
    features = np.zeros(get_feature_shape(dataset))
    progress_bar = tqdm(enumerate(iterate_human(dataset, wanted_human_id)), total=3*24)
    for i, (data, label, ids) in progress_bar:
        mne.set_log_level("WARNING")
        progress_bar.set_description(f"Processing (session_id, human_id, video_id) {str(ids)}")
        session_id, human_id, video_id = ids
        data -= human_params[human_id][0][:, None]
        data /= human_params[human_id][1][:, None]
        mne_data = convert_np2mne(data)
        mne_filtered_data = eeg_filter(mne_data, low_freq=0.5, high_freq=50)
        # get only first 5 seconds
        start = time.time()
        for i in range(0, min_time, 5):
            epoch_id = i // 5
            print(f"Processing {i} - {i+5} seconds")
            mne_crop_data = mne_filtered_data.copy()
            mne_crop_data = mne_crop_data.crop(tmin=i, tmax=i+5)
            for band_id, eeg_band in enumerate(split_abgd(mne_crop_data)):
                eeg_band_data = eeg_band.get_data()
                var = eeg_band_data.var(axis=-1)
                msv = np.mean((eeg_band_data**2), axis=-1)
                hjorth_mobility = mne_features.univariate.compute_hjorth_mobility(eeg_band_data)
                hjorth_complexity = mne_features.univariate.compute_hjorth_complexity(eeg_band_data)
                p2p = np.apply_along_axis(peak2peak, 1, eeg_band_data)
                aprox_entropy = np.apply_along_axis(antropy.app_entropy, 1, eeg_band_data)
                c0 = np.apply_along_axis(calculate_c0, 1, eeg_band_data)
                svd_entropy = mne_features.univariate.compute_svd_entropy(eeg_band_data)
                spectral_entropy = mne_features.univariate.compute_spect_entropy(Fs, eeg_band_data)
                permutation_entropy = np.apply_along_axis(antropy.perm_entropy, 1, eeg_band_data, normalize=True)
                # Ovo nesto zeza vraca dva broja ali ne znam sta su
                # kolmogorov_entropy = np.apply_along_axis(lambda x: EntropyHub.K2En(x)[0], 1, eeg_band_data)
                feature_list = [var, msv, hjorth_mobility, hjorth_complexity, p2p, aprox_entropy, c0, svd_entropy, spectral_entropy, permutation_entropy]
                features[session_id-1, human_id-1, video_id-1, epoch_id, band_id] = np.abs(np.stack(feature_list, axis=-1)).astype(np.float32)
        print(f"Time elapsed: {time.time() - start}")

    np.save(f"features{wanted_human_id}.npy", features)
