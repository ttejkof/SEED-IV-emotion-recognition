from utils import iterate_all, Dataset
import mne
import numpy as np
from mne.preprocessing import ICA, create_eog_epochs
dataset: Dataset = Dataset.get_dataset(reload=False)
montage = mne.channels.read_custom_montage(r'SEED-IV/channel_62_pos.locs')

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

for i, (data, label, ids) in enumerate(iterate_all(dataset)):
    print(data.shape, label, ids)
    mne_data = convert_np2mne(data)
    mne_data.filter(l_freq=1.0, h_freq=40.0)
    raw = mne_data
    ica = ICA(n_components=20, random_state=97, method='fastica')

    # Fit ICA on the raw data
    ica.fit(raw)

    eog_channel = raw.copy().pick_channels(['Fp1', 'Fp2']).get_data().mean(axis=0)
    info = mne.create_info(['EOG'], raw.info['sfreq'], ch_types=['eog'])
    synthetic_eog = mne.io.RawArray(eog_channel[np.newaxis, :], info)
    raw.add_channels([synthetic_eog], force_update_info=True)

    eog_inds, scores = ica.find_bads_eog(raw)


    # Plot the scores
    ica.plot_scores(scores)

    # Mark EOG components for exclusion
    ica.exclude = eog_inds

    # Optionally, you can visualize the ICA components to verify the automatic detection
    ica.plot_components()

    # Apply the ICA solution to the raw data
    raw_clean = ica.apply(raw.copy())

    # Plot the original and cleaned data to compare
    raw.plot(title='Original Data')
    raw_clean.plot(title='Cleaned Data')


    break