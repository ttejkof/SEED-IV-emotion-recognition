from .Dataset import Dataset
from typing import Iterator

def iterate_human(dataset:Dataset, human_id: int) -> Iterator:
    for session in dataset:
        for trial in session:
            if trial.human_id == human_id:
                for i, (eeg_recoding, label) in enumerate(trial):
                    yield eeg_recoding, label

def iterate_session(dataset:Dataset, session_id: int) -> Iterator:
    for session in dataset:
        if session.session_id == session_id:
            for trial in session:
                for i, (eeg_recoding, label) in enumerate(trial):
                    yield eeg_recoding, label

def iterate_label(dataset:Dataset, label: int) -> Iterator:
    for session in dataset:
        for trial in session:
            for i, (eeg_recoding, l) in enumerate(trial):
                if label == l:
                    yield eeg_recoding, label

def iterate_all(dataset:Dataset) -> Iterator:
    for session in dataset:
        for trial in session:
            for i, (eeg_recoding, label) in enumerate(trial):
                yield eeg_recoding, label
