from dataclasses import dataclass
import numpy as np
import mne
import mne_features
import antropy
from .config_utils import parse_config

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

def cacl_power_sum(eeg_band_data):
    [psd, psd_freq] = mne.time_frequency.psd_array_welch(eeg_band_data, sfreq = 200)
    power_sum = np.sum(psd, axis=-1)
    return power_sum

def calc_spectral_max(eeg_band_data):
    [psd, psd_freq] = mne.time_frequency.psd_array_welch(eeg_band_data, sfreq = 200)
    psd_max = np.max(psd, axis=-1)
    return psd_max

def calc_spectral_max_freq(eeg_band_data):
    [psd, psd_freq] = mne.time_frequency.psd_array_welch(eeg_band_data, sfreq = 200)
    psd_max_freq = psd_freq[np.argmax(psd, axis=-1)]
    return psd_max_freq

@dataclass
class Feature:
    name: str
    function: callable

feature_functions = {
    "var": lambda x: x.var(axis=-1),
    "msv": lambda x: np.mean((x**2), axis=-1),
    "hjorth_mobility": mne_features.univariate.compute_hjorth_mobility,
    "hjorth_complexity": mne_features.univariate.compute_hjorth_complexity,
    "p2p": lambda x: np.apply_along_axis(peak2peak, -1, x),
    "aprox_entropy": lambda x: np.apply_along_axis(antropy.app_entropy, -1, x),
    "c0": lambda x: np.apply_along_axis(calculate_c0, -1, x),
    # "svd_entropy": mne_features.univariate.compute_svd_entropy,
    # "spectral_entropy": lambda x: mne_features.univariate.compute_spect_entropy(Fs, x),
    # "permutation_entropy": lambda x: np.apply_along_axis(antropy.perm_entropy, -1, x, normalize=True),
    "power_sum": lambda x: np.apply_along_axis(cacl_power_sum, -1, x),
    "spectral_max": lambda x: np.apply_along_axis(calc_spectral_max, -1, x),
    "spectral_max_freq": lambda x: np.apply_along_axis(calc_spectral_max_freq, -1, x),
}

def get_feature_lambdas(config):
    return [feature_functions[feature_name] for feature_name in config if config[feature_name]]


if __name__ == "__main__":
    config = parse_config("config.yaml")
    feature_list = get_feature_lambdas(config["features"])
    print(feature_list)
    print(len(feature_list))
