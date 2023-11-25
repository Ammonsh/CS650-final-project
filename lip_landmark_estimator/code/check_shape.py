from pathlib import Path

import numpy as np


# Gets the paths of all the numpy files in the data directory
data_dir = Path("lip_landmark_estimator/data/lip_landmarks/")
lip_landmark_paths = [path for path in data_dir.glob("**/*.npy")]

for index, path in enumerate(sorted(lip_landmark_paths)):
    print(f"Processing video {index + 1}/{len(lip_landmark_paths)}")

    print(path)

    lip_landmarks = np.load(path)
    print(lip_landmarks.shape)
    print()