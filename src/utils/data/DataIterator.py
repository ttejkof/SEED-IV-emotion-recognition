from .Dataset import Dataset
from typing import Iterator

def iterate_human(dataset:Dataset, human_id: int) -> Iterator:
    for session in dataset:
        for trial in session:
            if trial.human_id == human_id:
                for i, (eeg_recoding, label) in enumerate(trial):
                    video_id = i
                    yield eeg_recoding, label, (session.session_id, trial.human_id, video_id)

def iterate_session(dataset:Dataset, session_id: int) -> Iterator:
    for session in dataset:
        if session.session_id == session_id:
            for trial in session:
                for i, (eeg_recoding, label) in enumerate(trial):
                    video_id = i
                    yield eeg_recoding, label, (session.session_id, trial.human_id, video_id)

def iterate_label(dataset:Dataset, label: int) -> Iterator:
    for session in dataset:
        for trial in session:
            for i, (eeg_recoding, l) in enumerate(trial):
                if label == l:
                    video_id = i
                    yield eeg_recoding, label, (session.session_id, trial.human_id, video_id)

def iterate_all(dataset:Dataset) -> Iterator:
    for session in dataset:
        for trial in session:
            for i, (eeg_recoding, label) in enumerate(trial):
                video_id = i
                yield eeg_recoding, label, (session.session_id, trial.human_id, video_id)

def get_all_human_ids(dataset:Dataset):
    return set([trial.human_id for session in dataset for trial in session])

def get_feature_shape(dataset:Dataset):
    # pod band_id mislim na alpha, beta, gamma, delta
    # (session_id, human_id, video_id, epoch_id, band_id, channel_id, feature_id)
    return (3, 15, 24, 8, 4, 62, 10)