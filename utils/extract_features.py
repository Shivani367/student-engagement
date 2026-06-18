import cv2
import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh

def extract_features(video_path):

    cap = cv2.VideoCapture(video_path)

    features = []

    with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True
    ) as face_mesh:

        while True:

            success, frame = cap.read()

            if not success:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = face_mesh.process(rgb)

            if results.multi_face_landmarks:

                landmarks = results.multi_face_landmarks[0].landmark

                left_eye = landmarks[33]
                right_eye = landmarks[263]
                mouth = landmarks[13]
                nose = landmarks[1]

                feature_vector = [
                    left_eye.x,
                    left_eye.y,
                    right_eye.x,
                    right_eye.y,
                    mouth.y,
                    nose.x,
                    nose.y
                ]

                features.append(feature_vector)

    cap.release()

    return np.array(features)