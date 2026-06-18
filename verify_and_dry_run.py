import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd

from utils.dataset import get_dataloaders
from models.stgcn import STGCNClassifier
from models.bigru import BiGRUClassifier
from models.convnext1d import ConvNeXt1DClassifier
from ensemble import EnsembleClassifier

def run_dry_run():
    print("=" * 60)
    print(" STARTING ONE-BATCH DRY-RUN VALIDATION")
    print("=" * 60)
    
    csv_path = "processed/labels.csv"
    processed_dir = "processed"
    
    # 1. Verify dataset files and labels
    print("\n[Step 1] Verifying dataset loading...")
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} does not exist!")
        sys.exit(1)
        
    df = pd.read_csv(csv_path)
    print(f"OK: labels.csv loaded successfully. Found {len(df)} samples.")
    print("Class distribution:\n", df['label'].value_counts())
    
    # Check if a few processed npy files exist
    for idx, row in df.head(5).iterrows():
        file_path = os.path.join(processed_dir, row['filename'])
        if not os.path.exists(file_path):
            print(f"Error: Processed file not found at {file_path}!")
            sys.exit(1)
    print("OK: Verified existence of processed .npy landmark files.")
    
    # 2. Load Dataloader (limit to 10 samples to test)
    print("\n[Step 2] Testing dataloader with limit_samples=10...")
    try:
        train_loader, val_loader = get_dataloaders(
            csv_path=csv_path,
            processed_dir=processed_dir,
            batch_size=4,
            val_split=0.2,
            limit_samples=10,
            seed=42
        )
        print("OK: Dataloaders created successfully.")
    except Exception as e:
        print(f"Error creating dataloaders: {e}")
        sys.exit(1)
        
    # Get a single batch of actual data
    print("\n[Step 3] Fetching one batch of actual data...")
    try:
        features, targets = next(iter(train_loader))
        print(f"OK: Batch loaded. Input features shape: {features.shape} (expected [batch, 300, 468, 2])")
        print(f"OK: Batch loaded. Targets shape: {targets.shape} (expected [batch])")
        assert features.shape[1:] == (300, 468, 2), f"Input shape mismatch! Got {features.shape}"
        assert len(targets.shape) == 1, f"Targets shape mismatch! Got {targets.shape}"
    except Exception as e:
        print(f"Error loading batch: {e}")
        sys.exit(1)
        
    # 3. Verify ST-GCN
    print("\n[Step 4] Verifying STGCNClassifier dimensions & forward/backward...")
    try:
        model = STGCNClassifier(num_classes=4)
        logits = model(features)
        print(f"OK: ST-GCN forward pass succeeded. Output shape: {logits.shape} (expected [batch, 4])")
        assert logits.shape == (features.size(0), 4), "ST-GCN output shape mismatch!"
        
        # Test backward pass
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        print("OK: ST-GCN backward pass & optimizer step succeeded.")
    except Exception as e:
        print(f"Error verifying ST-GCN: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
        
    # 4. Verify Bi-GRU
    print("\n[Step 5] Verifying BiGRUClassifier dimensions & forward/backward...")
    try:
        model = BiGRUClassifier(num_classes=4)
        logits = model(features)
        print(f"OK: Bi-GRU forward pass succeeded. Output shape: {logits.shape} (expected [batch, 4])")
        assert logits.shape == (features.size(0), 4), "Bi-GRU output shape mismatch!"
        
        # Test backward pass
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        print("OK: Bi-GRU backward pass & optimizer step succeeded.")
    except Exception as e:
        print(f"Error verifying Bi-GRU: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
        
    # 5. Verify ConvNeXt1D
    print("\n[Step 6] Verifying ConvNeXt1DClassifier dimensions & forward/backward...")
    try:
        model = ConvNeXt1DClassifier(num_classes=4)
        logits = model(features)
        print(f"OK: ConvNeXt1D forward pass succeeded. Output shape: {logits.shape} (expected [batch, 4])")
        assert logits.shape == (features.size(0), 4), "ConvNeXt1D output shape mismatch!"
        
        # Test backward pass
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        print("OK: ConvNeXt1D backward pass & optimizer step succeeded.")
    except Exception as e:
        print(f"Error verifying ConvNeXt1D: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
        
    # 6. Verify Ensemble soft voting
    print("\n[Step 7] Verifying Ensemble soft voting loading mock checkpoints...")
    try:
        checkpoint_dir = "checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save mock models
        torch.save(STGCNClassifier(num_classes=4).state_dict(), os.path.join(checkpoint_dir, "stgcn_best.pth"))
        torch.save(BiGRUClassifier(num_classes=4).state_dict(), os.path.join(checkpoint_dir, "bigru_best.pth"))
        torch.save(ConvNeXt1DClassifier(num_classes=4).state_dict(), os.path.join(checkpoint_dir, "convnext_best.pth"))
        
        # Initialize ensemble
        ensemble = EnsembleClassifier(
            stgcn_ckpt=os.path.join(checkpoint_dir, "stgcn_best.pth"),
            bigru_ckpt=os.path.join(checkpoint_dir, "bigru_best.pth"),
            convnext_ckpt=os.path.join(checkpoint_dir, "convnext_best.pth")
        )
        
        # Forward pass on ensemble
        probs = ensemble(features)
        print(f"OK: Ensemble forward pass succeeded. Output probabilities shape: {probs.shape} (expected [batch, 4])")
        assert probs.shape == (features.size(0), 4), "Ensemble output shape mismatch!"
        
        # Clean up mock checkpoints
        os.remove(os.path.join(checkpoint_dir, "stgcn_best.pth"))
        os.remove(os.path.join(checkpoint_dir, "bigru_best.pth"))
        os.remove(os.path.join(checkpoint_dir, "convnext_best.pth"))
        print("OK: Cleaned up mock checkpoints successfully.")
        
    except Exception as e:
        print(f"Error verifying Ensemble: {e}")
        sys.exit(1)

    print("\n" + "=" * 60)
    print(" ONE-BATCH DRY-RUN VALIDATION COMPLETED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == "__main__":
    run_dry_run()
