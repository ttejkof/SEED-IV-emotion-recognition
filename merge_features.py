
import numpy as np

file_names = [f"features{i}.npy" for i in range(1, 16)]

features = [np.load(file_name) for file_name in file_names]

all_features = np.zeros(features[0].shape)
for i  in range(1, 16):
    all_features[:, i-1] += features[i-1][:, i-1]

print(np.sum(all_features == 0))
np.save("all_features.npy", all_features)

