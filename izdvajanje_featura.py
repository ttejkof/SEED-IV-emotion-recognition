import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import mne
import pprint
from tqdm import tqdm
import pickle
import argparse

from utils.data import iterate_all, get_feature_shape, iterate_human
from utils.data import Dataset
from utils.config_utils import parse_config




human_params = pickle.load(open('human_params.pkl', 'rb'))
montage = mne.channels.read_custom_montage(r'SEED-IV/channel_62_pos.locs')




import scipy.signal

Fs = 200
min_time = 40 # OVo treba proveriti
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
                feature_list = [var, msv, hjorth_mobility, hjorth_complexity, p2p, aprox_entropy, c0, svd_entropy, spectral_entropy, permutation_entropy]
                features[session_id-1, human_id-1, video_id-1, epoch_id, band_id] = np.abs(np.stack(feature_list, axis=-1)).astype(np.float32)
        print(f"Time elapsed: {time.time() - start}")

    np.save(f"features{wanted_human_id}.npy", features)
