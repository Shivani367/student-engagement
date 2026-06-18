import torch
import torch.nn as nn

class BiGRUClassifier(nn.Module):
    """
    Bi-directional GRU model for student engagement classification.
    Input Shape: (batch, frames, 468, 2)
    Flattens landmarks to 936 features per frame, projects to a hidden dimension,
    passes through a Bi-GRU, and pools temporal outputs for classification.
    """
    def __init__(self, num_classes=4, input_dim=936, hidden_dim=128, num_layers=2, dropout=0.3):
        super().__init__()
        
        # Spatial embedding: project 936-dim flattened landmarks to hidden_dim
        self.embedding = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Bi-GRU layer
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        # Classifier head (takes concatenated mean and max pooling: 2 * hidden_size * 2 directions)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2 * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        # Input shape: (batch_size, frames, 468, 2)
        batch_size, seq_len, V, C = x.shape
        
        # Flatten landmarks per frame: (batch_size, frames, 936)
        x = x.view(batch_size, seq_len, -1)
        
        # Project spatial landmarks to embedding space: (batch_size, frames, hidden_dim)
        x_emb = self.embedding(x)
        
        # Pass through Bi-GRU: outputs shape (batch_size, frames, hidden_dim * 2)
        gru_out, _ = self.gru(x_emb)
        
        # Temporal pooling along the frame dimension (dim=1)
        # Combine mean pooling and max pooling to capture both global context and peak movements
        mean_pool = gru_out.mean(dim=1)  # (batch_size, hidden_dim * 2)
        max_pool, _ = gru_out.max(dim=1)   # (batch_size, hidden_dim * 2)
        
        # Concatenate poolings: (batch_size, hidden_dim * 4)
        pooled = torch.cat([mean_pool, max_pool], dim=-1)
        
        # Classify
        logits = self.fc(pooled)
        return logits
