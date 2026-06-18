import torch
import torch.nn as nn

class LayerNorm1D(nn.Module):
    """
    LayerNorm that operates on channels-first 1D tensors (batch_size, channels, length).
    """
    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps

    def forward(self, x):
        # Input shape: (N, C, L)
        # Convert to channels-last: (N, L, C)
        x = x.permute(0, 2, 1)
        
        # Calculate mean and variance along the channel dimension
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True, unbiased=False)
        
        # Normalize
        x = (x - mean) / torch.sqrt(var + self.eps)
        x = self.weight * x + self.bias
        
        # Convert back to channels-first: (N, C, L)
        return x.permute(0, 2, 1)

class ConvNeXt1DBlock(nn.Module):
    """
    ConvNeXt 1D Block.
    1. Depthwise 1D conv (kernel size 7, groups=channels)
    2. LayerNorm1D
    3. Pointwise conv (1x1 conv or Linear) to expand channels (4x)
    4. GELU activation
    5. Pointwise conv to project back
    6. Residual connection
    """
    def __init__(self, dim, drop_path=0.0):
        super().__init__()
        # Depthwise 1D conv
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=7, padding=3, groups=dim)
        # Layer Normalization
        self.norm = LayerNorm1D(dim)
        # Pointwise convs (inverted bottleneck)
        self.pwconv1 = nn.Conv1d(dim, 4 * dim, kernel_size=1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv1d(4 * dim, dim, kernel_size=1)
        
        self.drop_path = drop_path

    def forward(self, x):
        # Input shape: (N, C, L)
        residual = x
        
        # Depthwise convolution
        x = self.dwconv(x)
        x = self.norm(x)
        
        # Pointwise convs (inverted bottleneck)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        
        # Stochastic depth / drop path (optional, simplified as residual scaling)
        if self.drop_path > 0.0 and self.training:
            # Simple dropout on the residual path
            x = x * (torch.rand(x.size(0), 1, 1, device=x.device) > self.drop_path).float() / (1.0 - self.drop_path)
            
        return residual + x

class ConvNeXt1DClassifier(nn.Module):
    """
    ConvNeXt1D model for student engagement classification.
    Input Shape: (batch, frames, 468, 2)
    """
    def __init__(self, num_classes=4, input_dim=936, depths=[2, 2, 2], dims=[64, 128, 256], dropout=0.3):
        super().__init__()
        
        # Spatial projection: project flattened landmarks (936) to dims[0] (64)
        self.stem = nn.Sequential(
            nn.Conv1d(input_dim, dims[0], kernel_size=1),
            LayerNorm1D(dims[0])
        )
        
        self.stages = nn.ModuleList()
        self.downsample_layers = nn.ModuleList()
        
        # Build downsampling and stages
        for i in range(len(depths)):
            # Downsampling layer between stages
            if i > 0:
                downsample = nn.Sequential(
                    LayerNorm1D(dims[i-1]),
                    nn.Conv1d(dims[i-1], dims[i], kernel_size=2, stride=2)  # Downsample temporal length by 2
                )
                self.downsample_layers.append(downsample)
            else:
                self.downsample_layers.append(nn.Identity())
                
            # Stack of ConvNeXt blocks for this stage
            stage = nn.Sequential(
                *[ConvNeXt1DBlock(dim=dims[i]) for _ in range(depths[i])]
            )
            self.stages.append(stage)
            
        # Final classification layers
        self.norm = LayerNorm1D(dims[-1])
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        
        self.head = nn.Sequential(
            nn.Linear(dims[-1], 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # Input shape: (batch_size, frames, 468, 2)
        batch_size, seq_len, V, C = x.shape
        
        # Flatten landmarks per frame: (batch_size, frames, 936)
        x = x.view(batch_size, seq_len, -1)
        
        # Permute to channels-first: (batch_size, 936, frames)
        x = x.permute(0, 2, 1).contiguous()
        
        # Apply stem projection
        x = self.stem(x)
        
        # Pass through stage by stage
        for downsample, stage in zip(self.downsample_layers, self.stages):
            x = downsample(x)
            x = stage(x)
            
        # Final norm
        x = self.norm(x)
        
        # Average pool: (batch_size, dims[-1], 1)
        x = self.avgpool(x)
        
        # Flatten: (batch_size, dims[-1])
        x = x.view(batch_size, -1)
        
        # Classify
        logits = self.head(x)
        return logits
