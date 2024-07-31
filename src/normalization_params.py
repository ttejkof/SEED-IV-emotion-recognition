import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import mne 
import pprint
import pickle
from tqdm import tqdm

from utils.data import iterate_human, get_all_human_ids
from utils.data import Dataset

dataset: Dataset = Dataset.get_dataset(reload=False)

humans = get_all_human_ids(dataset)
humans

human_params = {}
progress_bar = tqdm(humans)
for human_id in progress_bar:
    big_array = []
    for i, (data, label, ids) in enumerate(iterate_human(dataset, human_id)):
        progress_bar.set_description(f"Processing {human_id}, (session_id, human_id, video_id) {str(ids)}")
        big_array.append(data)
    big_array = np.concatenate(big_array, axis=-1)
    params = big_array.mean(axis=-1), big_array.std(axis=-1)
    human_params[human_id] = params
    
pprint.pp(human_params)
print(human_params)

pickle.dump(human_params, open("human_params.pkl", "wb"))


