import cv2
import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh

def extract_features(video_path):

    cap = cv2.VideoCapture(video_path)
    features = []

    try:
        if not cap.isOpened():
            return np.empty((0, 468, 2), dtype=np.float32)

        with mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=False
        ) as face_mesh:

            while True:

                success, frame = cap.read()

                # End of file or corrupted/unreadable frame. Stop cleanly and
                # return the landmarks collected from previous valid frames.
                if not success or frame is None:
                    break

                try:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = face_mesh.process(rgb)
                except cv2.error:
                    continue

                # Skip frames where MediaPipe cannot detect a face.
                if not results.multi_face_landmarks:
                    continue

                landmarks = results.multi_face_landmarks[0].landmark

                frame_landmarks = [
                    [landmark.x, landmark.y]
                    for landmark in landmarks
                ]

                features.append(frame_landmarks)
    finally:
        cap.release()

    # Shape: (num_frames, 468, 2), where each frame stores all FaceMesh
    # landmarks as normalized [x, y] pairs. This layout can be fed to ST-GCN
    # as temporal graph input after any model-specific transpose/normalization.
    return np.asarray(features, dtype=np.float32).reshape(-1, 468, 2)
