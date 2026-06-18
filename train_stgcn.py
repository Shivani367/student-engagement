import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
from pathlib import Path

# Import our dataset module and ST-GCN model
from utils.dataset import get_dataloaders
from models.stgcn import STGCNClassifier

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_targets = []
    
    for features, targets in tqdm(dataloader, desc="Training Batches", leave=False):
        # Move tensors to device
        features = features.to(device)
        targets = targets.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        logits = model(features)
        loss = criterion(logits, targets)
        
        # Backward pass & optimize
        loss.backward()
        optimizer.step()
        
        # Track statistics
        running_loss += loss.item() * features.size(0)
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_targets.extend(targets.cpu().numpy())
        
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = accuracy_score(all_targets, all_preds)
    epoch_f1 = f1_score(all_targets, all_preds, average='weighted', zero_division=0)
    
    return epoch_loss, epoch_acc, epoch_f1

def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for features, targets in tqdm(dataloader, desc="Validation Batches", leave=False):
            features = features.to(device)
            targets = targets.to(device)
            
            logits = model(features)
            loss = criterion(logits, targets)
            
            running_loss += loss.item() * features.size(0)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(targets.cpu().numpy())
            
    val_loss = running_loss / len(dataloader.dataset)
    val_acc = accuracy_score(all_targets, all_preds)
    val_f1 = f1_score(all_targets, all_preds, average='weighted', zero_division=0)
    
    return val_loss, val_acc, val_f1

def main():
    parser = argparse.ArgumentParser(description="Train ST-GCN for Student Engagement Detection")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--csv_path", type=str, default="processed/labels.csv", help="Path to labels CSV")
    parser.add_argument("--processed_dir", type=str, default="processed", help="Directory of preprocessed landmark files")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory to save model checkpoints")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create checkpoints directory
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load Data
    print("Loading data splits...")
    train_loader, val_loader = get_dataloaders(
        csv_path=args.csv_path,
        processed_dir=args.processed_dir,
        batch_size=args.batch_size,
        val_split=0.2,
        max_seq_len=300,
        seed=args.seed
    )
    
    # Extract training labels to compute class weights
    train_labels = train_loader.dataset.labels
    
    # Calculate class weights for dealing with imbalance
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_labels),
        y=train_labels
    )
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
    print(f"Computed class weights: {class_weights}")
    
    # 2. Initialize Model
    print("Initializing ST-GCN model...")
    model = STGCNClassifier(num_classes=4, max_seq_len=300, num_nodes=468).to(device)
    
    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    # Training Loop
    best_val_f1 = 0.0
    best_epoch = 0
    
    print(f"Starting training for {args.epochs} epochs...")
    for epoch in range(1, args.epochs + 1):
        print(f"\n--- Epoch {epoch}/{args.epochs} ---")
        
        train_loss, train_acc, train_f1 = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_f1 = validate(model, val_loader, criterion, device)
        
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Train F1: {train_f1:.4f}")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f} | Val F1:   {val_f1:.4f}")
        
        # Save best checkpoint based on validation F1 score
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            checkpoint_path = checkpoint_dir / "stgcn_best.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': val_f1,
                'val_loss': val_loss
            }, checkpoint_path)
            print(f"New best model saved to {checkpoint_path} (Val F1: {val_f1:.4f})")
            
    print("\nTraining completed.")
    print(f"Best model was at Epoch {best_epoch} with Val F1: {best_val_f1:.4f}")
    
    # Save the final checkpoint
    final_path = checkpoint_dir / "stgcn_final.pth"
    torch.save(model.state_dict(), final_path)
    print(f"Final model saved to {final_path}")

if __name__ == "__main__":
    main()
