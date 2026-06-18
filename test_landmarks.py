from utils.extract_features import extract_features

video_path = r"C:\Users\rames\Desktop\student-engagement\dataset\DAiSEE\DataSet\Train\110002\1100021003\1100021003.avi"

data = extract_features(video_path)

print("Shape:", data.shape)
print("Datatype:", data.dtype)

print("First frame shape:", data[0].shape)