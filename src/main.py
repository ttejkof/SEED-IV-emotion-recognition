
from utils import iterate_all, Dataset
from utils.config_utils import parse_config
from utils.preprocessing import apply_ica, convert_np2mne, eeg_filter, split_abgd, normalize, interpolate
from utils.features import get_feature_lambdas
import time
import numpy as np
import mne
import concurrent.futures
import multiprocessing as mp
from tqdm import tqdm

def preprocess(signal, ids, config):
    normalized_signal = normalize(signal, ids, config)
    mne_signal = convert_np2mne(normalized_signal)
    filtered_signal = eeg_filter(mne_signal, **config['filter'])
    filtered_signal_wo_eog = interpolate(filtered_signal)
    return filtered_signal.get_data()

def split_epochs(signal, config):
    min_time = config["min_time"]
    Fs = config["Fs"]
    signal = signal[:, :min_time*Fs]

    epoch_time = config["epoch_time"]
    n_channels = signal.shape[0]
    epoch_samples = epoch_time * Fs
    total_samples = signal.shape[1]
    # Ensure that total_samples is divisible by epoch_samples
    if total_samples % epoch_samples != 0:
        raise ValueError("Total samples are not divisible by epoch samples. Adjust the epoch time or signal length.")
    
    n_epochs = total_samples // epoch_samples
    
    # Reshape the signal to (n_channels, n_epochs, epoch_samples) and then transpose to (n_epochs, n_channels, epoch_samples)
    reshaped_signal = signal.reshape(n_channels, n_epochs, epoch_samples)
    return reshaped_signal.transpose(1, 0, 2)

def process_trial(data, ids, config):
    feature_list = get_feature_lambdas(config["features"])
    session_id, human_id, video_id = ids
    preprocessed_signal = preprocess(data, ids, config)
    epochs = split_epochs(preprocessed_signal, config)
    bands = split_abgd(epochs)
    features = [feature(bands) for feature in feature_list]
    features = np.array(features)
    features = features.transpose(1, 2, 3, 0)
    np.save(f"features/{session_id}_{human_id}_{video_id}.npy", features)

def process_video_wrapper(args):
    return process_trial(*args)

def main():
    dataset: Dataset = Dataset.get_dataset(reload=False)
    config = parse_config("config.yaml")
    mne.set_log_level("WARNING")

    # for data, label, ids in iterate_all(dataset):
    #     process_trial(data, ids, config)
    #     break

    # Create a pool of worker processes
    pool = mp.Pool(10)
    total_items = 1080

    # Use tqdm to show progress bar
    progress_bar = tqdm(total=total_items)
    for result in pool.imap_unordered(process_video_wrapper, ((data, ids, config) for data, label, ids in iterate_all(dataset))):
        progress_bar.update(1)


    pass

if __name__ == "__main__":
    main()