from pathlib import Path

import cv2
import dlib
import numpy as np


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(
    "lip_landmark_estimator/data/pretrained_models/shape_predictor_68_face_landmarks.dat"
)

lip_landmark_indices = list(range(48, 68))
lip_landmark_list = []

cap = cv2.VideoCapture("lip_landmark_estimator/data/videos/7PwvGfs6Pok/50001.mp4")

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
        break

    face = faces[0]

    # Get the landmarks/parts for the face
    landmarks = predictor(gray, face)

    frame_landmarks = []

    # Iterate over each landmark and write its coordinates to the list
    for landmark in lip_landmark_indices:
        x = landmarks.part(landmark).x
        y = landmarks.part(landmark).y    
        frame_landmarks.append((x, y))

    lip_landmark_list.append(frame_landmarks)

    # # Display the frame
    # cv2.imshow("frame", frame)

    # # Exit when 'q' is pressed
    # if cv2.waitKey(30) & 0xFF == ord("q"):
    #     break

# Release the capture
cap.release()
cv2.destroyAllWindows()

lip_landmarks_array = np.array(lip_landmark_list)
# print(lip_landmarks_array.shape)
# print(lip_landmarks_array[0])