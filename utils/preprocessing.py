from mne.preprocessing import ICA
import mne
import pickle
import numpy as np
import scipy
def interpolate(raw):
    raw.info["bads"].extend(["Fp1", "Fpz", "Fp2", "AF3","AF4", "Fz", "F5", "F3","F1", "F7","FC5","FC3","FC1" ])  
    #sumoviti kanali - interpoliranje
    eeg_data_interp = raw.copy().interpolate_bads(reset_bads=True)
    return eeg_data_interp

def apply_ica(raw):
    
    ica = ICA(n_components=20, random_state=97, method='fastica')

    # Fit ICA on the raw data
    ica.fit(raw)

    eog_channel = raw.copy().pick_channels(['Fp1', 'Fp2']).get_data().mean(axis=0)
    info = mne.create_info(['EOG'], raw.info['sfreq'], ch_types=['eog'])
    synthetic_eog = mne.io.RawArray(eog_channel[np.newaxis, :], info)
    raw.add_channels([synthetic_eog], force_update_info=True)

    eog_inds, scores = ica.find_bads_eog(raw)

    # Mark EOG components for exclusion
    ica.exclude = eog_inds

    # Apply the ICA solution to the raw dataz1z
    raw_clean = ica.apply(raw.copy())
    raw_clean.drop_channels(['EOG'])

    return raw_clean

def eeg_filter(mne_data, low_freq, high_freq):
    return mne_data.copy().filter(low_freq, high_freq, 'eeg')

def split_abgd(signal):
    Fs = 100
    order = 5
    bands_range = [
        (8, 16),
        (16, 32),
        (32, 45),
        (4, 8),
    ]
    bands = []
    for band in bands_range:
        nyq = 0.5 * Fs # Nyquist frequency

        low, high = [x / nyq for x in band]
        b, a = scipy.signal.butter(order, [low, high], btype='band')
        bands.append(scipy.signal.filtfilt(b, a, signal, axis=-1))
    bands = np.array(bands)

    return bands.transpose(1, 0, 2, 3)

human_params = pickle.load(open('human_params.pkl', 'rb'))

def normalize(signal, ids, config):
    session_id, human_id, video_id = ids
    signal -= human_params[human_id][0][:, None]
    signal /= human_params[human_id][1][:, None]
    return signal


def convert_np2mne(data: np.ndarray):
    montage = mne.channels.read_custom_montage(r'SEED-IV/channel_62_pos.locs')
    n_channels = 62
    sampling_freq = 200  # in Hertz
    info = mne.create_info(n_channels, sfreq=sampling_freq)
    ch_names = ["Fp1", "Fpz", "Fp2", "AF3", "AF4", "F7", "F5", "F3", "F1", "Fz", "F2", "F4", "F6", "F8", "FT7", "FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6", "FT8", "T7", "C5", "C3", "C1", "Cz", "C2", "C4", "C6", "T8", "TP7", "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6", "TP8", "P7", "P5", "P3", "P1", "Pz", "P2", "P4", "P6", "P8", "PO7", "PO5", "PO3", "POz", "PO4", "PO6", "PO8", "CB1", "O1", "Oz", "O2", "CB2"]
    ch_types = ["eeg"]*62
    info = mne.create_info(ch_names, ch_types=ch_types, sfreq=sampling_freq)
    eeg = mne.io.RawArray(data/1e6, info)
    eeg.set_montage(montage)
    return eeg