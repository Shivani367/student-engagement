from utils.extract_features import extract_features

video_path = r"C:\Users\rames\Downloads\archive (3)\DAiSEE\DataSet\Train\110002\1100021003\1100021003.avi"

data = extract_features(video_path)

print("Shape:", data.shape)

if len(data) > 0:
    print(data[:5])