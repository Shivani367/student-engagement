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
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1, 2, 3], average='weighted', zero_division=0)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1, 2, 3], average='macro', zero_division=0)
    
    print(f"\n==================================================")
    print(f" Performance Report: {model_name}")
    print(f"==================================================")
    print(f"Accuracy:          {acc:.4f}")
    print(f"Weighted Precision: {precision:.4f}")
    print(f"Weighted Recall:    {recall:.4f}")
    print(f"Weighted F1 Score:  {f1:.4f}")
    print(f"Macro F1 Score:     {f1_macro:.4f}")
    
    # Detailed per-class precision, recall, f1 (always returns size 4)
    p_class, r_class, f1_class, support = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1, 2, 3], average=None, zero_division=0)
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
        'f1_macro': f1_macro,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'per_class': {
            'precision': p_class.tolist(),
            'recall': r_class.tolist(),
            'f1': f1_class.tolist(),
            'support': support.tolist()
        }
    }

def plot_save_learning_curves(history_path, model_name, output_dir):
    """
    Plots training and validation loss and accuracy curves from history JSON.
    """
    import json
    if not os.path.exists(history_path):
        print(f"History file not found at {history_path}. Skipping curves plotting for {model_name}.")
        return
        
    try:
        with open(history_path, 'r') as f:
            history = json.load(f)
            
        epochs = [h['epoch'] for h in history]
        train_losses = [h['train_loss'] for h in history]
        val_losses = [h['val_loss'] for h in history]
        train_accs = [h['train_acc'] for h in history]
        val_accs = [h['val_acc'] for h in history]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot Loss
        ax1.plot(epochs, train_losses, label='Train Loss', marker='o', color='#1f77b4')
        ax1.plot(epochs, val_losses, label='Val Loss', marker='s', color='#ff7f0e')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title(f'{model_name} - Loss Curve')
        ax1.legend()
        ax1.grid(True, linestyle='--', alpha=0.6)
        
        # Plot Accuracy
        ax2.plot(epochs, train_accs, label='Train Acc', marker='o', color='#2ca02c')
        ax2.plot(epochs, val_accs, label='Val Acc', marker='s', color='#d62728')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title(f'{model_name} - Accuracy Curve')
        ax2.legend()
        ax2.grid(True, linestyle='--', alpha=0.6)
        
        plt.suptitle(f'{model_name} Training History', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        output_path = Path(output_dir) / f"{model_name.lower().replace('-', '')}_curves.png"
        plt.savefig(output_path, dpi=300)
        plt.close()
        print(f"Saved learning curves plot to {output_path}")
    except Exception as e:
        print(f"Failed to plot learning curves for {model_name}: {e}")

def plot_save_confusion_matrix(y_true, y_pred, model_name, class_names, output_path):
    """
    Plots the confusion matrix and saves it as an image.
    """
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])
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
    parser.add_argument("--limit_samples", type=int, default=None, help="Limit number of dataset samples")
    
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
        seed=42,
        limit_samples=args.limit_samples
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
            
    # 5. Plot learning curves for individual models
    if "ST-GCN" in results_summary:
        plot_save_learning_curves(Path(args.stgcn_ckpt).parent / "stgcn_history.json", "ST-GCN", output_dir)
    if "Bi-GRU" in results_summary:
        plot_save_learning_curves(Path(args.bigru_ckpt).parent / "bigru_history.json", "Bi-GRU", output_dir)
    if "ConvNeXt1D" in results_summary:
        plot_save_learning_curves(Path(args.convnext_ckpt).parent / "convnext_history.json", "ConvNeXt1D", output_dir)
        
    # 6. Print final comparative summary and generate Markdown report
    if len(results_summary) > 0:
        print("\n" + "=" * 60)
        print(f"{'Model Name':<20} | {'Accuracy':<10} | {'Weighted F1':<12} | {'Macro F1':<10}")
        print("=" * 60)
        for model_name, metrics in results_summary.items():
            print(f"{model_name:<20} | {metrics['accuracy']:.4f}   | {metrics['f1_weighted']:.4f}      | {metrics['f1_macro']:.4f}")
        print("=" * 60 + "\n")
        
        # Generate Markdown Report
        report_path = output_dir / "final_evaluation_report.md"
        try:
            with open(report_path, "w") as f:
                f.write("# Final Evaluation Report - Student Engagement Detection\n\n")
                f.write("This report summarizes the performance of the ST-GCN, Bi-GRU, ConvNeXt1D, and Ensemble models on the validation split.\n\n")
                f.write("## 1. Overall Performance Comparison\n\n")
                f.write("| Model Name | Accuracy | Weighted F1-Score | Macro F1-Score | Weighted Precision | Weighted Recall |\n")
                f.write("| :--- | :--- | :--- | :--- | :--- | :--- |\n")
                for model_name, metrics in results_summary.items():
                    f.write(f"| {model_name} | {metrics['accuracy']:.4f} | {metrics['f1_weighted']:.4f} | {metrics['f1_macro']:.4f} | {metrics['precision_weighted']:.4f} | {metrics['recall_weighted']:.4f} |\n")
                f.write("\n")
                
                f.write("## 2. Detailed Per-Class Performance\n\n")
                for model_name, metrics in results_summary.items():
                    if 'per_class' in metrics:
                        f.write(f"### {model_name}\n\n")
                        f.write("| Class Name | Precision | Recall | F1-Score | Support |\n")
                        f.write("| :--- | :--- | :--- | :--- | :--- |\n")
                        pc = metrics['per_class']
                        for i, cname in enumerate(class_names):
                            f.write(f"| {cname} | {pc['precision'][i]:.4f} | {pc['recall'][i]:.4f} | {pc['f1'][i]:.4f} | {int(pc['support'][i])} |\n")
                        f.write("\n")
                
                f.write("## 3. Generated Plots and Visualizations\n\n")
                f.write("The following plots were generated and saved under the evaluation results directory:\n\n")
                f.write("### Confusion Matrices\n")
                for model_name in results_summary.keys():
                    mn_clean = model_name.lower().split(" ")[0].replace("-", "")
                    f.write(f"- **{model_name} Confusion Matrix**: `{mn_clean}_cm.png`\n")
                f.write("\n### Learning Curves\n")
                for model_name in results_summary.keys():
                    if model_name != "Ensemble":
                        mn_clean = model_name.lower().split(" ")[0].replace("-", "")
                        f.write(f"- **{model_name} Learning Curves**: `{mn_clean}_curves.png` (if training history was available)\n")
            print(f"Saved final evaluation report to {report_path}")
        except Exception as e:
            print(f"Failed to save evaluation report: {e}")

if __name__ == "__main__":
    main()
