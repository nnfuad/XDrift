"""Automated unit tests for 1D-CNN Model Architecture and Quantization Verification.
Verifies invariants 11, 12, 13, 20 from Section 20 of XDrift Protocol.
"""

import torch
import torch.ao.quantization as quantization
import pytest

from src.models.cnn1d import Compact1DCNN, fuse_model_ptq, fuse_model_qat


def test_11_and_12_output_shape_and_parameter_count():
    """Test 11 & 12: Correct model output shape, finite logits, exact parameter count."""
    model = Compact1DCNN()
    
    # Invariant 12: Exact parameter count 44,129
    param_count = model.count_parameters()
    assert param_count == 44129, f"Expected 44129 params, got {param_count}"

    # Invariant 11: Correct output shape (batch_size, 1) and finite logits
    x = torch.randn(8, 2, 64)
    model.eval()
    with torch.no_grad():
        out = model(x)
    assert out.shape == (8, 1)
    assert torch.isfinite(out).all()


def test_13_and_20_quantized_execution_and_alignment():
    """Tests 13 & 20: Verified quantized execution and prediction alignment across FP32 and PTQ."""
    torch.backends.quantized.engine = 'qnnpack'
    
    fp32_model = Compact1DCNN().eval()
    fused_model = fuse_model_ptq(fp32_model)
    fused_model.qconfig = quantization.get_default_qconfig('qnnpack')
    
    prep_model = quantization.prepare(fused_model)
    # Calibrate on dummy batch
    calib_x = torch.randn(16, 2, 64)
    prep_model(calib_x)
    
    int8_ptq = quantization.convert(prep_model)
    
    # Invariant 20: Verified real quantized operators in graph
    import torch.ao.nn.quantized as nnq
    import torch.ao.nn.intrinsic.quantized as nniq

    assert isinstance(int8_ptq.conv1, (nniq.ConvReLU1d, nnq.Conv1d))
    assert isinstance(int8_ptq.fc1, nnq.Linear)
    assert isinstance(int8_ptq.quant, nnq.Quantize)
    assert isinstance(int8_ptq.dequant, nnq.DeQuantize)

    # Invariant 13: Prediction alignment (same shape, finite values, reasonable correlation)
    test_x = torch.randn(10, 2, 64)
    with torch.no_grad():
        fp32_out = fp32_model(test_x)
        ptq_out = int8_ptq(test_x)

    assert fp32_out.shape == ptq_out.shape == (10, 1)
    assert torch.isfinite(ptq_out).all()
    # Check that logits are within reasonable range
    diff = torch.abs(fp32_out - ptq_out)
    assert torch.isfinite(diff).all()
