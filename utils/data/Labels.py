import numpy as np

class Labels:
    @staticmethod
    def get_session_labels(id: int) -> np.ndarray:
        session_labels = [[1,2,3,0,2,0,0,1,0,1,2,1,1,1,2,3,2,2,3,3,0,3,0,3],
        [2,1,3,0,0,2,0,2,3,3,2,3,2,0,1,1,2,1,0,3,0,1,3,1],
        [1,2,2,1,3,3,3,1,1,2,1,0,2,3,3,0,2,3,0,0,2,0,1,0]]
        session_labels = np.array(session_labels)
        return session_labels[id]

    @staticmethod
    def label_names(id: int):
        label_names = ['Neutral', 'Sad', 'Fear', 'Happy']
        return label_names[id]
