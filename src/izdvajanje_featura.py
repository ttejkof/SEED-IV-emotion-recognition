import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import mne
import pprint
from tqdm import tqdm
import pickle
import argparse
import time

from torch.utils.data import DataLoader

from utils.data import iterate_all, get_feature_shape, iterate_human, IterateHuman
from utils.data import Dataset
from utils.config_utils import parse_config
from utils.preprocessing import convert_np2mne, eeg_filter, split_abgd

human_params = pickle.load(open('human_params.pkl', 'rb'))
montage = mne.channels.read_custom_montage(r'SEED-IV/channel_62_pos.locs')
dataset: Dataset = Dataset.get_dataset(reload=False)

import scipy.signal

Fs = 200
min_time = 40

# Function to compute specific feature
def compute_specific_feature(eeg_band_data, feature_name):
    if feature_name == "var":
        return eeg_band_data.var(axis=-1)
    elif feature_name == "msv":
        return np.mean((eeg_band_data**2), axis=-1)
    elif feature_name == "hjorth_mobility":
        return mne_features.univariate.compute_hjorth_mobility(eeg_band_data)
    elif feature_name == "hjorth_complexity":
        return mne_features.univariate.compute_hjorth_complexity(eeg_band_data)
    elif feature_name == "p2p":
        return np.apply_along_axis(peak2peak, 1, eeg_band_data)
    elif feature_name == "approx_entropy":
        return np.apply_along_axis(antropy.app_entropy, 1, eeg_band_data)
    elif feature_name == "c0":
        return np.apply_along_axis(calculate_c0, 1, eeg_band_data)
    elif feature_name == "svd_entropy":
        return mne_features.univariate.compute_svd_entropy(eeg_band_data)
    elif feature_name == "spectral_entropy":
        return mne_features.univariate.compute_spect_entropy(Fs, eeg_band_data)
    elif feature_name == "permutation_entropy":
        return np.apply_along_axis(antropy.perm_entropy, 1, eeg_band_data, normalize=True)
    else:
        raise ValueError(f"Unknown feature: {feature_name}")

# Set the feature you want to extract
feature_name = "var"


for wanted_human_id in range(1, 16):
    print(f"Extracting feature {feature_name} for human {wanted_human_id}")
    torch_dataset = IterateHuman(dataset, wanted_human_id)
    dataLoader = DataLoader(torch_dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=lambda x: x[0])
    features = np.zeros(get_feature_shape(dataset))
    progress_bar = tqdm(enumerate(dataLoader), total=3*24)
    for i, (data, label, ids) in progress_bar:
        mne.set_log_level("WARNING")
        progress_bar.set_description(f"Processing (session_id, human_id, video_id) {str(ids)}")
        session_id, human_id, video_id = ids
        data -= human_params[human_id][0][:, None]
        data /= human_params[human_id][1][:, None]
        mne_data = convert_np2mne(data)
        mne_filtered_data = eeg_filter(mne_data, low_freq=0.5, high_freq=50)
        start = time.time()
        for i in range(0, min_time, 5):
            epoch_id = i // 5
            mne_crop_data = mne_filtered_data.copy().crop(tmin=i, tmax=i+5)
            mne_crop_data = mne_crop_data.get_data()
            for band_id, eeg_band in enumerate(split_abgd(mne_crop_data)):
                specific_feature = compute_specific_feature(eeg_band, feature_name)
                features[session_id-1, human_id-1, video_id-1, epoch_id, band_id] = np.abs(specific_feature).astype(np.float32)
        # print(f"Time elapsed: {time.time() - start}")

    np.save(f"features_{feature_name}_{wanted_human_id}.npy", features)
