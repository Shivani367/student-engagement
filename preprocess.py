import csv
from pathlib import Path

import numpy as np
from tqdm import tqdm

from utils.extract_features import extract_features


DATASET_DIR = Path("dataset/DAiSEE")
LABELS_PATH = DATASET_DIR / "Labels" / "TrainLabels.csv"
TRAIN_VIDEOS_DIR = DATASET_DIR / "DataSet" / "Train"
PROCESSED_DIR = Path("processed")
PROCESSED_LABELS_PATH = PROCESSED_DIR / "labels.csv"
FAILURES_LOG_PATH = PROCESSED_DIR / "failures.log"


def build_video_index():
    return {
        video_path.name: video_path
        for video_path in TRAIN_VIDEOS_DIR.rglob("*.avi")
    }


def locate_video(clip_id, video_index):
    clip_path = Path(clip_id)
    clip_name = clip_path.name
    clip_stem = clip_path.stem
    subject_id = clip_stem[:6]

    expected_path = TRAIN_VIDEOS_DIR / subject_id / clip_stem / clip_name
    if expected_path.exists():
        return expected_path

    return video_index.get(clip_name)


def log_failure(log_file, clip_id, reason):
    log_file.write(f"{clip_id},{reason}\n")


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    video_index = build_video_index()

    with LABELS_PATH.open("r", newline="") as labels_file:
        rows = list(csv.DictReader(labels_file))

    with (
        PROCESSED_LABELS_PATH.open("w", newline="") as processed_labels_file,
        FAILURES_LOG_PATH.open("w", newline="") as failures_log_file,
    ):
        labels_writer = csv.writer(processed_labels_file)
        labels_writer.writerow(["filename", "label"])
        failures_log_file.write("clip_id,reason\n")

        for row in tqdm(rows, desc="Extracting landmarks"):
            clip_id = row["ClipID"]
            engagement_label = row["Engagement"]
            video_path = locate_video(clip_id, video_index)

            if video_path is None:
                log_failure(failures_log_file, clip_id, "video_not_found")
                continue

            try:
                landmarks = extract_features(str(video_path))
            except Exception as exc:
                log_failure(failures_log_file, clip_id, f"extract_failed:{type(exc).__name__}")
                continue

            # Corrupted videos or clips without detected faces produce no ST-GCN
            # frames. Landmark tensors are saved as (num_frames, 468, 2) float32.
            if landmarks.shape[0] == 0:
                log_failure(failures_log_file, clip_id, "no_landmarks")
                continue

            landmarks = landmarks.astype(np.float32, copy=False)
            output_filename = f"{Path(clip_id).stem}.npy"
            output_path = PROCESSED_DIR / output_filename

            np.save(output_path, landmarks)
            labels_writer.writerow([output_filename, engagement_label])


if __name__ == "__main__":
    main()
