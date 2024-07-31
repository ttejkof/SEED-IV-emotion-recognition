import numpy as np

def load_dataset(filename: str) -> np.ndarray:
    data = np.load(filename)
    labels_ids = [[1,2,3,0,2,0,0,1,0,1,2,1,1,1,2,3,2,2,3,3,0,3,0,3],
        [2,1,3,0,0,2,0,2,3,3,2,3,2,0,1,1,2,1,0,3,0,1,3,1],
        [1,2,2,1,3,3,3,1,1,2,1,0,2,3,3,0,2,3,0,0,2,0,1,0]]
    labels = np.zeros(data.shape[:-3])
    groups = np.zeros(data.shape[:-3])
    for session in range(data.shape[0]):
        for human in range(data.shape[1]):
            for video in range(data.shape[2]):
                labels[session,human,video] = labels_ids[session][video]
                groups[session,human,video] = human

    return data, labels, groups

def flatten_data(features, labels, groups):
    num_sessions, num_humans, num_videos, num_epochs, num_bands, num_channels, num_features = features.shape
    flattened_features = features.reshape(num_sessions, num_humans, num_videos, num_epochs, -1)

    # Ensure labels shape matches the first four dimensions
    flattened_labels = labels.reshape(num_sessions, num_humans, num_videos, num_epochs)
    flattened_groups = groups.reshape(num_sessions, num_humans, num_videos, num_epochs)

    # Flatten the arrays along the first four dimensions
    flattened_features = flattened_features.reshape(-1, num_bands * num_channels * num_features)

    return flattened_features, flattened_labels.flatten(), flattened_groups.flatten()

def select_emotions(data, labels, groups, em1, em2):
    labels = labels.copy()
    indices = np.logical_or((labels == em1) , (labels == em2))
    labels[labels == em1] = 0
    labels[labels == em2] = 1
    return data[indices], labels[indices], groups[indices]

def train_test_split(data, labels, groups, id):
    indices = groups == id
    return data[~indices], labels[~indices], data[indices], labels[indices]