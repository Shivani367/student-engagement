import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from pathlib import Path

class EngagementDataset(Dataset):
    """
    PyTorch Dataset for Online Student Engagement Detection.
    Loads facial landmark sequences extracted by MediaPipe FaceMesh.
    Input Shape: (num_frames, 468, 2) -> padded/truncated to (max_seq_len, 468, 2)
    """
    def __init__(self, filenames, labels, processed_dir, max_seq_len=300):
        self.filenames = list(filenames)
        self.labels = list(labels)
        self.processed_dir = Path(processed_dir)
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        label = self.labels[idx]
        file_path = self.processed_dir / filename
        
        try:
            # Memory mapping for fast on-the-fly loading
            landmarks = np.load(file_path, mmap_mode='r')
            T, V, C = landmarks.shape
            
            # Handle sequence length variations
            if T < self.max_seq_len:
                pad_width = ((0, self.max_seq_len - T), (0, 0), (0, 0))
                # Pad with zeros
                landmarks = np.pad(landmarks, pad_width, mode='constant', constant_values=0.0)
            elif T > self.max_seq_len:
                # Truncate
                landmarks = landmarks[:self.max_seq_len]
        except Exception as e:
            # Fallback in case of loading error: return zeros
            landmarks = np.zeros((self.max_seq_len, 468, 2), dtype=np.float32)
            
        # Convert to float32 tensor
        features = torch.tensor(landmarks, dtype=torch.float32)
        # Convert label to long tensor
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        return features, label_tensor

def get_dataloaders(csv_path, processed_dir, batch_size=32, val_split=0.2, max_seq_len=300, seed=42):
    """
    Reads the labels CSV and returns stratified train and validation PyTorch DataLoaders.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Labels file not found at {csv_path}")
        
    df = pd.read_csv(csv_path)
    
    # Stratified split to handle severe class imbalance
    train_df, val_df = train_test_split(
        df, 
        test_size=val_split, 
        random_state=seed, 
        stratify=df['label']
    )
    
    print(f"Dataset Split Summary:")
    print(f"Total samples: {len(df)}")
    print(f"Training samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")
    print("\nTraining Class Distribution:")
    print(train_df['label'].value_counts())
    print("\nValidation Class Distribution:")
    print(val_df['label'].value_counts())
    
    # Create dataset instances
    train_dataset = EngagementDataset(
        filenames=train_df['filename'],
        labels=train_df['label'],
        processed_dir=processed_dir,
        max_seq_len=max_seq_len
    )
    
    val_dataset = EngagementDataset(
        filenames=val_df['filename'],
        labels=val_df['label'],
        processed_dir=processed_dir,
        max_seq_len=max_seq_len
    )
    
    # Create DataLoader instances
    # Using num_workers=0 to avoid multiprocessing issues in Windows sandboxes
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )
    
    return train_loader, val_loader
