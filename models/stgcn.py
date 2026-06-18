import numpy as np
import torch
import torch.nn as nn

class FaceMeshGraph:
    """
    Builds the normalized adjacency matrix for MediaPipe FaceMesh (468 landmarks).
    Uses FACEMESH_TESSELATION connections.
    """
    def __init__(self, num_nodes=468):
        self.num_nodes = num_nodes
        self.adj = self.build_adjacency()

    def build_adjacency(self):
        # Start with identity matrix (self-connections)
        adj = np.eye(self.num_nodes, dtype=np.float32)
        try:
            import mediapipe as mp
            connections = mp.solutions.face_mesh.FACEMESH_TESSELATION
            for u, v in connections:
                if u < self.num_nodes and v < self.num_nodes:
                    adj[u, v] = 1.0
                    adj[v, u] = 1.0
        except Exception as e:
            # Fallback: simple chain graph if mediapipe is not available
            print("Warning: Could not load MediaPipe FaceMesh connections. Using chain fallback.")
            for i in range(self.num_nodes - 1):
                adj[i, i + 1] = 1.0
                adj[i + 1, i] = 1.0
        
        # Symmetrically normalize: D^-1/2 * A * D^-1/2
        row_sum = adj.sum(axis=1)
        d_inv_sqrt = np.zeros_like(row_sum)
        mask = row_sum > 0
        d_inv_sqrt[mask] = np.power(row_sum[mask], -0.5)
        d_mat_inv_sqrt = np.diag(d_inv_sqrt)
        
        adj_normalized = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)
        return torch.tensor(adj_normalized, dtype=torch.float32)

class SpatialGraphConv(nn.Module):
    """
    Spatial Graph Convolutional Layer.
    Computes Y = (A * E) * (X * W) where E is a learnable edge importance weight.
    """
    def __init__(self, in_channels, out_channels, adj, num_nodes=468):
        super().__init__()
        self.num_nodes = num_nodes
        # Register adj as a buffer so it moves with the model to device
        self.register_buffer('adj', adj.clone())
        
        # Learnable edge importance weight (initialized to 1s)
        self.edge_importance = nn.Parameter(torch.ones(num_nodes, num_nodes))
        
        # Conv2D to perform X * W over channels
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        # Input shape: (N, C_in, T, V)
        # Perform 1x1 convolution to change channels: (N, C_out, T, V)
        x_proj = self.conv(x)
        
        # Scale adjacency matrix by learnable edge importance
        A = self.adj * self.edge_importance
        
        # Graph convolution: multiply projected features by adjacency matrix on the right
        # X_proj has shape (N, C_out, T, V), A has shape (V, V)
        # Output shape: (N, C_out, T, V)
        out = torch.matmul(x_proj, A)
        return out

class STGCNBlock(nn.Module):
    """
    Spatial-Temporal Graph Convolutional Block.
    Contains:
    1. Spatial GCN
    2. Temporal 1D Convolution
    """
    def __init__(self, in_channels, out_channels, adj, t_kernel_size=9, stride=1, num_nodes=468, dropout=0.1):
        super().__init__()
        # Spatial Graph Convolution
        self.gcn = SpatialGraphConv(in_channels, out_channels, adj, num_nodes=num_nodes)
        
        # Temporal Convolution (TCN)
        # Convolves along the time dimension T. We use kernel size (t_kernel_size, 1) and stride (stride, 1)
        padding = ((t_kernel_size - 1) // 2, 0)
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels, 
                out_channels, 
                kernel_size=(t_kernel_size, 1), 
                stride=(stride, 1), 
                padding=padding
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout)
        )
        
        # Residual connection
        if stride != 1 or in_channels != out_channels:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.residual = nn.Identity()

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # Input shape: (N, C_in, T, V)
        res = self.residual(x)
        
        # Spatial GCN
        x = self.gcn(x)
        
        # Temporal GCN
        x = self.tcn(x)
        
        # Residual combination
        return self.relu(x + res)

class STGCNClassifier(nn.Module):
    """
    ST-GCN model for online student engagement classification.
    Takes input landmarks of shape (batch, frames, 468, 2)
    Outputs logits of shape (batch, 4)
    """
    def __init__(self, num_classes=4, max_seq_len=300, num_nodes=468):
        super().__init__()
        
        # Initialize FaceMeshGraph adjacency
        graph = FaceMeshGraph(num_nodes=num_nodes)
        adj = graph.adj
        
        # First layer: Project coordinate features (x, y) to a larger channel space
        # Input has shape (N, T, V, C) -> transpose to (N, C, T, V)
        self.input_bn = nn.BatchNorm2d(2)
        
        # Stacked Spatial-Temporal blocks
        # Downsample time dimension via strided temporal convolutions
        self.layer1 = STGCNBlock(2, 32, adj, t_kernel_size=9, stride=2, num_nodes=num_nodes)   # T: 300 -> 150
        self.layer2 = STGCNBlock(32, 64, adj, t_kernel_size=9, stride=2, num_nodes=num_nodes)  # T: 150 -> 75
        self.layer3 = STGCNBlock(64, 128, adj, t_kernel_size=9, stride=2, num_nodes=num_nodes) # T: 75 -> 38
        self.layer4 = STGCNBlock(128, 256, adj, t_kernel_size=9, stride=1, num_nodes=num_nodes)
        
        # Global pooling: mean over time (T) and nodes (V)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classification head
        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # Input shape: (N, T, V, C) = (batch_size, 300, 468, 2)
        # Transpose to (N, C, T, V) to align with standard ST-GCN format
        x = x.permute(0, 3, 1, 2).contiguous()
        
        # Batch normalization over input coordinates
        x = self.input_bn(x)
        
        # Pass through ST-GCN layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Pool to shape: (N, 256, 1, 1)
        x = self.pool(x)
        
        # Flatten to (N, 256)
        x = x.view(x.size(0), -1)
        
        # Classify
        logits = self.fc(x)
        return logits
