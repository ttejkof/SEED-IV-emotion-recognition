import numpy as np
from scipy.io import loadmat
import os
import pickle
from utils.data import Dataset, Labels
from utils.data import iterate_human, iterate_session, iterate_label, iterate_all

if __name__ == "__main__":
    dataset: Dataset = Dataset.get_dataset(reload=False)
    # for session in dataset:
    #     for trial in session:
    #         print(f"Session {session.session_id} Trial {trial.human_id} Trial_path {trial.path}", end="\n")
    #         for i, (eeg_recoding, label) in enumerate(trial):
    #             print(f"EEG Data for video {trial.videos[i]} i={i}", end="\t")
    #             print(f"EEG Data shape: {eeg_recoding.shape}", end="\t")
    #             print(f"Label: {Labels.label_names(label)}")

    print("Iterating over all eeg recordings of specific human")
    count = 0
    for i, (eeg_recoding, label) in enumerate(iterate_human(dataset, 1)):
        # print(f"EEG Data shape: {eeg_recoding.shape}", end="\t")
        # print(f"Label: {Labels.label_names(label)}")
        count += 1
    print(f"Count = {count} for human 1") # Should be 24*3 = 72
    
    count = 0
    print("Iterating over all eeg recordings of specific session")
    for i, (eeg_recoding, label) in enumerate(iterate_session(dataset, 1)):
        # print(f"EEG Data shape: {eeg_recoding.shape}", end="\t")
        # print(f"Label: {Labels.label_names(label)}")
        count += 1
    print(f"Count = {count} for session 1") # Should be 24*15 = 360
    
    count = 0
    print("Iterating over all eeg recordings of specific label")
    for i, (eeg_recoding, label) in enumerate(iterate_label(dataset, 1)):
        # print(f"EEG Data shape: {eeg_recoding.shape}", end="\t")
        # print(f"Label: {Labels.label_names(label)}")
        count += 1
    print(f"Count = {count} for label 1") # Should be 15*18 = 270

    count = 0
    print("Iterating over all eeg recordings")
    for i, (eeg_recoding, label) in enumerate(iterate_all(dataset)):
        # print(f"EEG Data shape: {eeg_recoding.shape}", end="\t")
        # print(f"Label: {Labels.label_names(label)}")
        count += 1
    print(f"Count = {count}") # Should be 24*15*3 = 1080