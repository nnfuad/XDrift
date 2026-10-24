"""Post-Training Quantization (PTQ) module for XDrift protocol (Section 9.1).

Converts trained FP32 Compact1DCNN to static INT8 using the QNNPACK engine.
Uses representative training data for calibration (never validation or test data).
Verifies real quantized operators in the converted graph.
"""

import os
import json
import time
from typing import Dict, Any, Tuple

import torch
import torch.ao.quantization as quantization
import torch.ao.nn.quantized as nnq
import torch.ao.nn.intrinsic.quantized as nniq

from src.models.cnn1d import Compact1DCNN, fuse_model_ptq
from src.training.train import load_dataset


def calibrate_model(model: torch.nn.Module, calib_loader: torch.utils.data.DataLoader) -> None:
    """Calibrate observer statistics on representative training batches."""
    model.eval()
    with torch.no_grad():
        for bx, _ in calib_loader:
            model(bx)


def run_ptq(
    fp32_checkpoint_path: str,
    train_npz_path: str,
    output_model_path: str,
    calibration_samples: int = 1000,
    seed: int = 42
) -> Dict[str, Any]:
    """Execute static INT8 Post-Training Quantization."""
    torch.backends.quantized.engine = 'qnnpack'

    # 1. Load trained FP32 checkpoint
    ckpt = torch.load(fp32_checkpoint_path, map_location='cpu')
    fp32_model = Compact1DCNN()
    fp32_model.load_state_dict(ckpt['state_dict'])
    fp32_model.eval()

    # 2. Fuse modules (Conv1d + BatchNorm1d + ReLU)
    fused_model = fuse_model_ptq(fp32_model)
    fused_model.qconfig = quantization.get_default_qconfig('qnnpack')

    # 3. Prepare for PTQ
    prep_model = quantization.prepare(fused_model)

    # 4. Calibrate using training data only
    X_train, _, y_train, _, _ = load_dataset(train_npz_path)
    
    # Balanced calibration sampling from training set
    torch.manual_seed(seed)
    idx_vpn = (y_train.squeeze() == 1).nonzero(as_tuple=True)[0]
    idx_nonvpn = (y_train.squeeze() == 0).nonzero(as_tuple=True)[0]

    n_half = calibration_samples // 2
    perm_vpn = idx_vpn[torch.randperm(len(idx_vpn))[:min(n_half, len(idx_vpn))]]
    perm_nonvpn = idx_nonvpn[torch.randperm(len(idx_nonvpn))[:min(n_half, len(idx_nonvpn))]]
    calib_idx = torch.cat([perm_vpn, perm_nonvpn])

    calib_X = X_train[calib_idx]
    calib_dataset = torch.utils.data.TensorDataset(calib_X, y_train[calib_idx])
    calib_loader = torch.utils.data.DataLoader(calib_dataset, batch_size=64, shuffle=False)

    print(f"Calibrating PTQ model with {len(calib_idx)} training samples ({len(perm_vpn)} VPN, {len(perm_nonvpn)} Non-VPN)...")
    t0 = time.time()
    calibrate_model(prep_model, calib_loader)
    calib_time = time.time() - t0

    # 5. Convert to quantized integer graph
    int8_model = quantization.convert(prep_model)
    int8_model.eval()

    # 6. Verify real quantized execution
    verified_ops = {}
    for name, m in int8_model.named_modules():
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

    # 7. Save quantized model
    os.makedirs(os.path.dirname(output_model_path), exist_ok=True)
    payload = {
        'model_type': 'INT8-PTQ',
        'backend': 'qnnpack',
        'calibration_samples': len(calib_idx),
        'calib_vpn_count': len(perm_vpn),
        'calib_nonvpn_count': len(perm_nonvpn),
        'calibration_time_seconds': round(calib_time, 2),
        'quantized_operators': verified_ops,
        'state_dict': int8_model.state_dict(),
        'validation_threshold': ckpt.get('validation_threshold', 0.5)
    }
    torch.save(payload, output_model_path)
    print(f"Saved INT8-PTQ model to {output_model_path} with verified operators: {list(verified_ops.keys())}")

    return payload
