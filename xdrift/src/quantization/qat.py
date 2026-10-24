"""Quantization-Aware Training (QAT) module for XDrift protocol (Section 9.2).

Fine-tunes trained FP32 Compact1DCNN with simulated fake quantization,
selects best checkpoint on validation data, and converts to deployed INT8 model.
Verifies real quantized operators in the converted graph.
"""

import os
import json
import time
from typing import Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.ao.quantization as quantization
import torch.ao.nn.quantized as nnq
import torch.ao.nn.intrinsic.quantized as nniq
from torch.utils.data import TensorDataset, DataLoader

from src.models.cnn1d import Compact1DCNN, fuse_model_qat
from src.training.train import load_dataset, compute_metrics, set_seed


def run_qat(
    fp32_checkpoint_path: str,
    train_npz_path: str,
    val_npz_path: str,
    output_model_path: str,
    epochs: int = 10,
    lr: float = 1e-4,
    batch_size: int = 64,
    seed: int = 42,
    device: str = "cpu"
) -> Dict[str, Any]:
    """Execute Quantization-Aware Training and conversion."""
    torch.backends.quantized.engine = 'qnnpack'
    set_seed(seed)
    dev = torch.device(device)

    # 1. Load trained FP32 checkpoint
    ckpt = torch.load(fp32_checkpoint_path, map_location='cpu')
    fp32_model = Compact1DCNN()
    fp32_model.load_state_dict(ckpt['state_dict'])

    # 2. Fuse modules for QAT
    fused_model = fuse_model_qat(fp32_model)
    fused_model.qconfig = quantization.get_default_qat_qconfig('qnnpack')

    # 3. Prepare model for QAT (inserts fake-quantization nodes)
    qat_model = quantization.prepare_qat(fused_model)
    qat_model.to(dev)

    # 4. Load dataset
    X_train, _, y_train, _, _ = load_dataset(train_npz_path)
    X_val, _, y_val, _, _ = load_dataset(val_npz_path)

    pos_weight = torch.tensor([ckpt.get('pos_weight', 1.0)]).to(dev)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(qat_model.parameters(), lr=lr, weight_decay=1e-5)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=False)

    print(f"Starting QAT fine-tuning for {epochs} epochs...")
    best_val_loss = float('inf')
    best_state = None
    best_epoch = -1

    t0 = time.time()
    for ep in range(1, epochs + 1):
        qat_model.train()
        train_loss = 0.0
        train_batches = 0

        for bx, by in train_loader:
            bx, by = bx.to(dev), by.to(dev)
            optimizer.zero_grad()
            logits = qat_model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_batches += 1

        train_loss /= max(1, train_batches)

        # Validation
        qat_model.eval()
        val_loss = 0.0
        val_batches = 0
        all_logits = []
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(dev), by.to(dev)
                logits = qat_model(bx)
                loss = criterion(logits, by)
                val_loss += loss.item()
                val_batches += 1
                all_logits.append(logits.cpu().numpy())

        val_loss /= max(1, val_batches)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = ep
            best_state = {k: v.cpu().clone() for k, v in qat_model.state_dict().items()}

        print(f"  QAT Epoch {ep:02d}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")

    qat_duration = time.time() - t0

    # 5. Convert best QAT model to deployed INT8 model
    qat_model.load_state_dict(best_state)
    qat_model.eval()
    int8_qat_model = quantization.convert(qat_model)

    # 6. Verify real quantized operators
    verified_ops = {}
    for name, m in int8_qat_model.named_modules():
        mod_type = type(m).__name__
        if isinstance(m, (nniq.ConvReLU1d, nnq.Conv1d)):
            verified_ops[name] = {'type': mod_type, 'scale': float(m.scale), 'zero_point': int(m.zero_point)}
        elif isinstance(m, nnq.Linear):
            verified_ops[name] = {'type': mod_type, 'scale': float(m.scale), 'zero_point': int(m.zero_point)}
        elif isinstance(m, nnq.Quantize):
            verified_ops[name] = {'type': mod_type, 'scale': float(m.scale.item()), 'zero_point': int(m.zero_point.item())}

    assert 'conv1' in verified_ops, "Conv1 was not quantized!"
    assert 'fc1' in verified_ops, "Linear1 was not quantized!"
    assert 'quant' in verified_ops, "Quantize stub was not quantized!"

    # 7. Save converted INT8-QAT model
    os.makedirs(os.path.dirname(output_model_path), exist_ok=True)
    payload = {
        'model_type': 'INT8-QAT',
        'backend': 'qnnpack',
        'qat_epochs': epochs,
        'best_epoch': best_epoch,
        'best_val_loss': float(best_val_loss),
        'qat_duration_seconds': round(qat_duration, 2),
        'quantized_operators': verified_ops,
        'state_dict': int8_qat_model.state_dict(),
        'validation_threshold': ckpt.get('validation_threshold', 0.5)
    }
    torch.save(payload, output_model_path)
    print(f"Saved INT8-QAT model to {output_model_path} with verified operators: {list(verified_ops.keys())}")

    return payload
