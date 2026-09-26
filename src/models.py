import os
import torch
import torch.nn as nn
import torch.nn.functional as F

class ACFACNN(nn.Module):
    """
    CNN Architecture for Acoustic Fault Detection and Classification.
    Matches state_dict structure of acfa_cnn_best.pth checkpoint.
    """
    def __init__(self, num_classes: int = 8, dropout_rate: float = 0.5):
        super(ACFACNN, self).__init__()
        
        # Conv Block 1
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        # Conv Block 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        # Conv Block 3
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        
        # Fully Connected Layers
        self.fc1 = nn.Linear(128 * 4 * 4, 128)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: (batch_size, 1, n_mels, time_frames)
        if x.dim() == 3:
            x = x.unsqueeze(1)
            
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)  # Flatten: (batch_size, 2048)
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def load_acfa_model(checkpoint_path: str = None, device: str = 'cpu') -> ACFACNN:
    """
    Helper function to instantiate ACFACNN model and load weights from checkpoint.
    """
    model = ACFACNN(num_classes=8)
    if checkpoint_path and os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        model.eval()
    model.to(device)
    return model
