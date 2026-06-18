import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.stgcn import STGCNClassifier
from models.bigru import BiGRUClassifier
from models.convnext1d import ConvNeXt1DClassifier

class EnsembleClassifier(nn.Module):
    """
    Ensemble model combining ST-GCN, Bi-GRU, and ConvNeXt1D models using soft voting.
    Loads trained checkpoints and computes averaged probability predictions.
    """
    def __init__(self, stgcn_ckpt=None, bigru_ckpt=None, convnext_ckpt=None, device="cpu"):
        super().__init__()
        self.device = torch.device(device)
        self.models = nn.ModuleDict()
        
        # Load ST-GCN if checkpoint provided
        if stgcn_ckpt and os.path.exists(stgcn_ckpt):
            print(f"Loading ST-GCN checkpoint from {stgcn_ckpt}...")
            stgcn_model = STGCNClassifier(num_classes=4)
            checkpoint = torch.load(stgcn_ckpt, map_location=self.device)
            # Support loading both state dict directly or from checkpoint dict
            state_dict = checkpoint.get('model_state_dict', checkpoint)
            stgcn_model.load_state_dict(state_dict)
            stgcn_model.eval()
            self.models['stgcn'] = stgcn_model.to(self.device)
        else:
            print("ST-GCN model NOT loaded (no valid checkpoint provided).")
            
        # Load Bi-GRU if checkpoint provided
        if bigru_ckpt and os.path.exists(bigru_ckpt):
            print(f"Loading Bi-GRU checkpoint from {bigru_ckpt}...")
            bigru_model = BiGRUClassifier(num_classes=4)
            checkpoint = torch.load(bigru_ckpt, map_location=self.device)
            state_dict = checkpoint.get('model_state_dict', checkpoint)
            bigru_model.load_state_dict(state_dict)
            bigru_model.eval()
            self.models['bigru'] = bigru_model.to(self.device)
        else:
            print("Bi-GRU model NOT loaded (no valid checkpoint provided).")
            
        # Load ConvNeXt1D if checkpoint provided
        if convnext_ckpt and os.path.exists(convnext_ckpt):
            print(f"Loading ConvNeXt1D checkpoint from {convnext_ckpt}...")
            convnext_model = ConvNeXt1DClassifier(num_classes=4)
            checkpoint = torch.load(convnext_ckpt, map_location=self.device)
            state_dict = checkpoint.get('model_state_dict', checkpoint)
            convnext_model.load_state_dict(state_dict)
            convnext_model.eval()
            self.models['convnext'] = convnext_model.to(self.device)
        else:
            print("ConvNeXt1D model NOT loaded (no valid checkpoint provided).")
            
        if len(self.models) == 0:
            raise RuntimeError("Ensemble initialized with 0 models! Please provide at least one valid checkpoint.")

    def forward(self, x):
        # Input shape: (batch, frames, 468, 2)
        # We perform soft voting: average predictions across all models
        probs_sum = None
        count = 0
        
        with torch.no_grad():
            for name, model in self.models.items():
                logits = model(x)
                # Compute softmax to get probabilities
                probs = F.softmax(logits, dim=-1)
                
                if probs_sum is None:
                    probs_sum = probs
                else:
                    probs_sum = probs_sum + probs
                count += 1
                
        # Average probability distribution
        avg_probs = probs_sum / count
        return avg_probs

    def predict(self, x):
        """
        Predicts classes for the input tensor.
        Returns predicted class labels (tensor).
        """
        avg_probs = self.forward(x)
        preds = torch.argmax(avg_probs, dim=-1)
        return preds
