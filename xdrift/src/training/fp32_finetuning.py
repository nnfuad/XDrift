"""Unquantized FP32 Fine-Tuning Control Baseline module.

Fine-tunes the converged FP32 Compact1DCNN for 5 identical epochs with AdamW
using the exact same learning rate, weight decay, loss weighting, and seed as QAT,
but in full 32-bit floating-point precision (WITHOUT fake quantization).
Disentangles gradient descent parameter shift from INT8 quantization constraints.
"""

import os
import json
import time
from typing import Dict, Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from src.models.cnn1d import Compact1DCNN

def set_seed(seed: int = 42) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_fp32_finetuning(
    fp32_checkpoint_path: str = "xdrift/artifacts/checkpoints/fp32_seed_42.pt",
    train_npz_path: str = "xdrift/data/processed/train_data.npz",
    val_npz_path: str = "xdrift/data/processed/val_data.npz",
    output_model_path: str = "xdrift/artifacts/checkpoints/fp32_finetuned_seed_42.pt",
    epochs: int = 5,
    lr: float = 1e-4,
    weight_decay: float = 1e-5,
    batch_size: int = 64,
    seed: int = 42,
    device: str = "cpu"
) -> Dict[str, Any]:
    """Execute unquantized FP32 fine-tuning control baseline."""
    set_seed(seed)
    dev = torch.device(device)

    # 1. Load trained FP32 checkpoint
    ckpt = torch.load(fp32_checkpoint_path, map_location='cpu')
    model = Compact1DCNN()
    model.load_state_dict(ckpt['state_dict'])
    model.to(dev)

    # 2. Load dataset
    d_train = np.load(train_npz_path)
    X_train = torch.tensor(d_train['X'], dtype=torch.float32)
    y_train = torch.tensor(d_train['y'], dtype=torch.float32).unsqueeze(1)

    d_val = np.load(val_npz_path)
    X_val = torch.tensor(d_val['X'], dtype=torch.float32)
    y_val = torch.tensor(d_val['y'], dtype=torch.float32).unsqueeze(1)

    pos_weight = torch.tensor([ckpt.get('pos_weight', 30.3756)]).to(dev)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=False)

    print(f"Starting FP32 Unquantized Fine-Tuning for {epochs} epochs on {device}...")
    best_val_loss = float('inf')
    best_state = None
    best_epoch = -1

    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_batches = 0

        for bx, by in train_loader:
            bx, by = bx.to(dev), by.to(dev)
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_batches += 1

        train_loss /= max(1, train_batches)

        # Validation
        model.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(dev), by.to(dev)
                logits = model(bx)
                loss = criterion(logits, by)
                val_loss += loss.item()
                val_batches += 1

        val_loss /= max(1, val_batches)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = ep
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        print(f"  FP32-FT Epoch {ep:02d}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")

    duration = time.time() - t0

    os.makedirs(os.path.dirname(output_model_path), exist_ok=True)
    payload = {
        'model_type': 'FP32-FineTuned',
        'epochs': epochs,
        'best_epoch': best_epoch,
        'best_val_loss': float(best_val_loss),
        'duration_seconds': round(duration, 2),
        'state_dict': best_state,
        'pos_weight': float(pos_weight.item()),
        'lr': lr,
        'weight_decay': weight_decay,
        'seed': seed,
        'validation_threshold': ckpt.get('validation_threshold', 0.900)
    }
    torch.save(payload, output_model_path)
    print(f"Saved FP32 Fine-Tuned checkpoint to {output_model_path}")
    return payload


if __name__ == "__main__":
    run_fp32_finetuning()
