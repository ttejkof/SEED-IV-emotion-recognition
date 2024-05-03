import os
import numpy as np

from .Trial import Trial
from .Labels import Labels

class Session:
    def __init__(self, path) -> None:
        print(f"Loading Session {path}")
        self.path = path
        self.session_id: int = int(path.split('/')[-1]) - 1
        self.session_labels: np.ndarray = Labels.get_session_labels(self.session_id)
        self.trials: list[Trial] = self.get_trials()
    
    def get_trials(self):
        trials = []
        trials_paths = os.listdir(self.path)
        trials_paths = filter(lambda x: x.endswith('.mat'), trials_paths)
        trials_paths = sorted(trials_paths, key=lambda x: int(x.split('_')[0]))
        for file in trials_paths:
            print(f"Loading Trial {file}")
            trial_num = int(file.split('_')[0]) - 1
            label = self.session_labels[trial_num]
            trials.append(Trial(os.path.join(self.path, file)))
        return trials

    def __getitem__(self, idx):
        return self.trials[idx]
    
    def __len__(self):
        return len(self.trials)