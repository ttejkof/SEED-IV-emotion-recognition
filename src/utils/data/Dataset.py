import pickle
import os
from .Session import Session

class Dataset:
    @staticmethod
    def get_dataset(reload: bool = False):
        dataset_pickle_path = "SEED-IV/dataset.pkl"
        if not reload and os.path.exists(dataset_pickle_path):
            dataset = Dataset.load_pickle(dataset_pickle_path)
        else:
            dataset = Dataset()
            dataset.save(dataset_pickle_path)
        return dataset

    @staticmethod
    def load_pickle(path):
        with open(path, 'rb') as f:
            return pickle.load(f)

    def __init__(self, path='SEED-IV/eeg_raw_data') -> None:
        self.raw_eeg_path = path
        self.sessions = []
        print(os.listdir("./"))
        sessions = os.listdir(self.raw_eeg_path)
        sessions = sorted(sessions, key=lambda x: int(x))
        for session in sessions:
            self.sessions.append(Session(os.path.join(self.raw_eeg_path, session)))

    def __getitem__(self, idx):
        return self.sessions[idx]
    
    def __len__(self):
        return len(self.sessions)
    
    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)