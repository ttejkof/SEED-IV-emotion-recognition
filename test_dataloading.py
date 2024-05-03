import numpy as np
from scipy.io import loadmat
import os
import pickle
from utils.data import Dataset, Session, Trial, Labels


if __name__ == "__main__":
    dataset: Dataset = Dataset.get_dataset(reload=False)
    for session in dataset:
        for trial in session:
            print(f"Session {session.session_id} Trial {trial.human_id} Trial_path {trial.path}", end="\n")
            for i, (eeg_recoding, label) in enumerate(trial):
                print(f"EEG Data for video {trial.videos[i]} i={i}", end="\t")
                print(f"EEG Data shape: {eeg_recoding.shape}", end="\t")
                print(f"Label: {Labels.label_names(label)}")
