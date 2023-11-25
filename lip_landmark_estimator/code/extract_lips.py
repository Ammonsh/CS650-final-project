from pathlib import Path

import cv2
import dlib
import numpy as np

# Create face detector and landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(
    "lip_landmark_estimator/data/pretrained_models/shape_predictor_68_face_landmarks.dat"
)

# Indices of the landmarks corresponding to the lips
lip_landmark_indices = list(range(48, 68))



# Gets the paths of all the videos in the data directory
data_dir = Path("lip_landmark_estimator/data/videos/")
video_paths = [path for path in data_dir.glob("**/*.mp4")]

# Create output directory if it doesn't exist
output_dir = Path("lip_landmark_estimator/data/lip_landmarks/")
output_dir.mkdir(parents=True, exist_ok=True)

for index, path in enumerate(sorted(video_paths)):
    print(f"Processing video {index + 1}/{len(video_paths)}")
    cap = cv2.VideoCapture(str(path))
    print(path)

    # Create list to store all lip landmarks from all frames
    lip_landmark_list = []

    while cap.isOpened():
        # Read a frame from the video capture
        read, frame = cap.read()
        if not read:
            break

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = detector(gray)
        if len(faces) > 1:
            print("More than one face detected!")
            continue
        elif len(faces) == 0:
            print("No faces detected!")
            continue

        # print(faces)

        for face in faces:
            landmarks = predictor(gray, face)

            # Creates list to store all lip landmarks for the current frame
            frame_landmark_list = []

            # Appends the x and y coordinates of each lip landmark
            for landmark in lip_landmark_indices:
                x = landmarks.part(landmark).x
                y = landmarks.part(landmark).y
                # draw a circle on the lip landmarks
                cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)
                
                frame_landmark_list.append((x, y))

        lip_landmark_list.append(frame_landmark_list)

        cv2.imshow("Frame", frame)
        if cv2.waitKey(30) == ord("q"):
            break
    
    lip_landmark_array = np.array(lip_landmark_list)
    print(lip_landmark_array.shape)
    
    numpy_file = f"{path.parent.name}_{path.stem}.npy"

    np.save(Path(output_dir / numpy_file), lip_landmark_array)

cap.release()
cv2.destroyAllWindows()