"""Compact 1D-CNN Architecture for Encrypted Traffic Classification (Section 7).

Input: (batch, 2, 64)
- Block 1: Conv1d(2, 32, kernel_size=5, padding=2) -> BatchNorm1d(32) -> ReLU -> MaxPool1d(2)
- Block 2: Conv1d(32, 64, kernel_size=5, padding=2) -> BatchNorm1d(64) -> ReLU -> MaxPool1d(2)
- Block 3: Conv1d(64, 128, kernel_size=3, padding=1) -> BatchNorm1d(128) -> ReLU -> AdaptiveAvgPool1d(1)
- Classifier: Flatten -> Linear(128, 64) -> ReLU -> Dropout(0.3) -> Linear(64, 1)

Total parameters: exactly 44,129.
"""

import torch
import torch.nn as nn
import torch.ao.quantization as quantization


class Compact1DCNN(nn.Module):
    def __init__(self, in_channels: int = 2, num_classes: int = 1, dropout_rate: float = 0.3):
        super().__init__()
        self.quant = quantization.QuantStub()
        
        # Block 1
        self.conv1 = nn.Conv1d(in_channels, 32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(32)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(2)
        
        # Block 2
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(2)
        
        # Block 3
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.relu3 = nn.ReLU()
        self.gap = nn.AdaptiveAvgPool1d(1)
        
        # Classifier
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128, 64)
        self.relu4 = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(64, num_classes)
        
        self.dequant = quantization.DeQuantStub()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.quant(x)
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.gap(self.relu3(self.bn3(self.conv3(x))))
        x = self.flatten(x)
        x = self.dropout(self.relu4(self.fc1(x)))
        x = self.fc2(x)
        return self.dequant(x)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


def fuse_model_ptq(model: Compact1DCNN) -> Compact1DCNN:
    """Fuse Conv+BN+ReLU modules for PTQ."""
    model.eval()
    return quantization.fuse_modules(model, [
        ['conv1', 'bn1', 'relu1'],
        ['conv2', 'bn2', 'relu2'],
        ['conv3', 'bn3', 'relu3']
    ], inplace=False)


def fuse_model_qat(model: Compact1DCNN) -> Compact1DCNN:
    """Fuse Conv+BN+ReLU modules for QAT."""
    model.train()
    return quantization.fuse_modules_qat(model, [
        ['conv1', 'bn1', 'relu1'],
        ['conv2', 'bn2', 'relu2'],
        ['conv3', 'bn3', 'relu3']
    ], inplace=False)
