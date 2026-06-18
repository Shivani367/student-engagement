import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, ConfusionMatrixDisplay
from pathlib import Path

# Import our dataset, models, and ensemble
from utils.dataset import get_dataloaders
from models.stgcn import STGCNClassifier
from models.bigru import BiGRUClassifier
from models.convnext1d import ConvNeXt1DClassifier
from ensemble import EnsembleClassifier

def evaluate_model(model, dataloader, device, is_ensemble=False):
    """
    Evaluates a single model or ensemble on the dataloader.
    Returns y_true and y_pred list.
    """
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for features, targets in tqdm(dataloader, desc="Evaluating", leave=False):
            features = features.to(device)
            
            if is_ensemble:
                # ensemble output is averaged probabilities
                probs = model(features)
                preds = torch.argmax(probs, dim=-1).cpu().numpy()
            else:
                logits = model(features)
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                
            all_preds.extend(preds)
            all_targets.extend(targets.numpy())
            
    return np.array(all_targets), np.array(all_preds)

def calculate_metrics(y_true, y_pred, model_name, class_names):
    """
    Computes accuracy, precision, recall, f1 and prints the report.
    Returns metrics dict.
    """
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    
    print(f"\n==================================================")
    print(f" Performance Report: {model_name}")
    print(f"==================================================")
    print(f"Accuracy:          {acc:.4f}")
    print(f"Weighted Precision: {precision:.4f}")
    print(f"Weighted Recall:    {recall:.4f}")
    print(f"Weighted F1 Score:  {f1:.4f}")
    print(f"Macro F1 Score:     {f1_macro:.4f}")
    
    # Detailed per-class precision, recall, f1
    p_class, r_class, f1_class, support = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
    print("\nPer-Class Metrics:")
    print(f"{'Class':<20} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Support':<10}")
    print("-" * 65)
    for i, name in enumerate(class_names):
        print(f"{name:<20} | {p_class[i]:.4f}     | {r_class[i]:.4f}   | {f1_class[i]:.4f}   | {int(support[i])}")
    print("==================================================\n")
    
    return {
        'accuracy': acc,
        'precision_weighted': precision,
        'recall_weighted': recall,
        'f1_weighted': f1,
        'f1_macro': f1_macro
    }

def plot_save_confusion_matrix(y_true, y_pred, model_name, class_names, output_path):
    """
    Plots the confusion matrix and saves it as an image.
    """
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(cmap=plt.cm.Blues, ax=ax, values_format='d')
    plt.title(f"Confusion Matrix - {model_name}")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved confusion matrix plot to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate Individual and Ensemble Models")
    parser.add_argument("--stgcn_ckpt", type=str, default="checkpoints/stgcn_best.pth", help="Path to ST-GCN checkpoint")
    parser.add_argument("--bigru_ckpt", type=str, default="checkpoints/bigru_best.pth", help="Path to Bi-GRU checkpoint")
    parser.add_argument("--convnext_ckpt", type=str, default="checkpoints/convnext_best.pth", help="Path to ConvNeXt1D checkpoint")
    parser.add_argument("--csv_path", type=str, default="processed/labels.csv", help="Path to labels CSV")
    parser.add_argument("--processed_dir", type=str, default="processed", help="Directory of preprocessed landmark files")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--output_dir", type=str, default="evaluation_results", help="Directory to save plots and reports")
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device for evaluation: {device}")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Define classes in DAiSEE
    class_names = ["0: Disengaged", "1: Low Engaged", "2: Engaged", "3: Highly Engaged"]
    
    # Load Validation Dataloader (uses the same split logic and seed=42)
    print("Loading validation dataloader...")
    _, val_loader = get_dataloaders(
        csv_path=args.csv_path,
        processed_dir=args.processed_dir,
        batch_size=args.batch_size,
        val_split=0.2,
        seed=42
    )
    
    # Store performance comparison
    results_summary = {}
    
    # 1. Evaluate ST-GCN
    if os.path.exists(args.stgcn_ckpt):
        print("\nLoading and evaluating ST-GCN...")
        try:
            model = STGCNClassifier(num_classes=4)
            checkpoint = torch.load(args.stgcn_ckpt, map_location=device)
            state_dict = checkpoint.get('model_state_dict', checkpoint)
            model.load_state_dict(state_dict)
            model = model.to(device)
            
            y_true, y_pred = evaluate_model(model, val_loader, device, is_ensemble=False)
            metrics = calculate_metrics(y_true, y_pred, "ST-GCN", class_names)
            results_summary["ST-GCN"] = metrics
            plot_save_confusion_matrix(y_true, y_pred, "ST-GCN", class_names, output_dir / "stgcn_cm.png")
        except Exception as e:
            print(f"Failed to evaluate ST-GCN: {e}")
    else:
        print(f"ST-GCN checkpoint not found at {args.stgcn_ckpt}. Skipping.")
        
    # 2. Evaluate Bi-GRU
    if os.path.exists(args.bigru_ckpt):
        print("\nLoading and evaluating Bi-GRU...")
        try:
            model = BiGRUClassifier(num_classes=4)
            checkpoint = torch.load(args.bigru_ckpt, map_location=device)
            state_dict = checkpoint.get('model_state_dict', checkpoint)
            model.load_state_dict(state_dict)
            model = model.to(device)
            
            y_true, y_pred = evaluate_model(model, val_loader, device, is_ensemble=False)
            metrics = calculate_metrics(y_true, y_pred, "Bi-GRU", class_names)
            results_summary["Bi-GRU"] = metrics
            plot_save_confusion_matrix(y_true, y_pred, "Bi-GRU", class_names, output_dir / "bigru_cm.png")
        except Exception as e:
            print(f"Failed to evaluate Bi-GRU: {e}")
    else:
        print(f"Bi-GRU checkpoint not found at {args.bigru_ckpt}. Skipping.")
        
    # 3. Evaluate ConvNeXt1D
    if os.path.exists(args.convnext_ckpt):
        print("\nLoading and evaluating ConvNeXt1D...")
        try:
            model = ConvNeXt1DClassifier(num_classes=4)
            checkpoint = torch.load(args.convnext_ckpt, map_location=device)
            state_dict = checkpoint.get('model_state_dict', checkpoint)
            model.load_state_dict(state_dict)
            model = model.to(device)
            
            y_true, y_pred = evaluate_model(model, val_loader, device, is_ensemble=False)
            metrics = calculate_metrics(y_true, y_pred, "ConvNeXt1D", class_names)
            results_summary["ConvNeXt1D"] = metrics
            plot_save_confusion_matrix(y_true, y_pred, "ConvNeXt1D", class_names, output_dir / "convnext_cm.png")
        except Exception as e:
            print(f"Failed to evaluate ConvNeXt1D: {e}")
    else:
        print(f"ConvNeXt1D checkpoint not found at {args.convnext_ckpt}. Skipping.")
        
    # 4. Evaluate Ensemble
    available_ckpts = [c for c in [args.stgcn_ckpt, args.bigru_ckpt, args.convnext_ckpt] if os.path.exists(c)]
    if len(available_ckpts) > 0:
        print("\nEvaluating Ensemble Model...")
        try:
            ensemble_model = EnsembleClassifier(
                stgcn_ckpt=args.stgcn_ckpt if os.path.exists(args.stgcn_ckpt) else None,
                bigru_ckpt=args.bigru_ckpt if os.path.exists(args.bigru_ckpt) else None,
                convnext_ckpt=args.convnext_ckpt if os.path.exists(args.convnext_ckpt) else None,
                device=device
            )
            
            y_true, y_pred = evaluate_model(ensemble_model, val_loader, device, is_ensemble=True)
            metrics = calculate_metrics(y_true, y_pred, "Ensemble (Soft Voting)", class_names)
            results_summary["Ensemble"] = metrics
            plot_save_confusion_matrix(y_true, y_pred, "Ensemble", class_names, output_dir / "ensemble_cm.png")
        except Exception as e:
            print(f"Failed to evaluate Ensemble: {e}")
            
    # 5. Print final comparative summary
    if len(results_summary) > 0:
        print("\n" + "=" * 60)
        print(f"{'Model Name':<20} | {'Accuracy':<10} | {'Weighted F1':<12} | {'Macro F1':<10}")
        print("=" * 60)
        for model_name, metrics in results_summary.items():
            print(f"{model_name:<20} | {metrics['accuracy']:.4f}   | {metrics['f1_weighted']:.4f}      | {metrics['f1_macro']:.4f}")
        print("=" * 60 + "\n")

if __name__ == "__main__":
    main()
