"""Hardware, Latency, and Resource Benchmarking Module (Section 16).

Measures:
- Model artifact size on disk (bytes / KB / MB)
- Parameter count and memory footprint
- Inference latency at Batch Size = 1 (primary edge inference condition)
- Inference latency at Batch Size = 64
- Warmup cycles and statistical dispersion (median, IQR, p95, p99)
- Quantized operator coverage analysis
"""

import os
import time
import json
from typing import Dict, List, Any

import numpy as np
import torch

from src.models.cnn1d import Compact1DCNN
from src.evaluation.evaluate_models import load_int8_model


def measure_inference_latency(
    model: torch.nn.Module,
    batch_size: int = 1,
    n_warmup: int = 50,
    n_runs: int = 500,
    device: str = "cpu"
) -> Dict[str, float]:
    """Benchmark inference latency with warmup and statistical dispersion reporting."""
    model.eval()
    dev = torch.device(device)
    dummy_x = torch.randn(batch_size, 2, 64, device=dev)

    # 1. Warmup runs
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(dummy_x)

    # 2. Timing runs
    timings = []
    with torch.no_grad():
        for _ in range(n_runs):
            t0 = time.perf_counter()
            _ = model(dummy_x)
            t1 = time.perf_counter()
            timings.append((t1 - t0) * 1000.0)  # milliseconds

    timings_arr = np.array(timings)
    median_ms = float(np.median(timings_arr))
    mean_ms = float(np.mean(timings_arr))
    std_ms = float(np.std(timings_arr))
    p25 = float(np.percentile(timings_arr, 25))
    p75 = float(np.percentile(timings_arr, 75))
    iqr_ms = p75 - p25
    p95_ms = float(np.percentile(timings_arr, 95))
    p99_ms = float(np.percentile(timings_arr, 99))
    min_ms = float(np.min(timings_arr))
    max_ms = float(np.max(timings_arr))
    throughput = float((batch_size / (mean_ms / 1000.0))) if mean_ms > 0 else 0.0

    return {
        'batch_size': batch_size,
        'runs': n_runs,
        'median_latency_ms': median_ms,
        'mean_latency_ms': mean_ms,
        'std_latency_ms': std_ms,
        'iqr_latency_ms': iqr_ms,
        'p95_latency_ms': p95_ms,
        'p99_latency_ms': p99_ms,
        'min_latency_ms': min_ms,
        'max_latency_ms': max_ms,
        'throughput_samples_per_sec': throughput
    }


def benchmark_all_models(
    fp32_checkpoint_path: str,
    ptq_model_path: str,
    qat_model_path: str,
    output_path: str
) -> Dict[str, Any]:
    """Benchmark artifact sizes, parameter counts, and latency across FP32, PTQ, and QAT."""
    # 1. Load models
    # FP32
    fp32_ckpt = torch.load(fp32_checkpoint_path, map_location='cpu')
    fp32_model = Compact1DCNN().eval()
    fp32_model.load_state_dict(fp32_ckpt['state_dict'])

    # PTQ
    ptq_payload = torch.load(ptq_model_path, map_location='cpu')
    ptq_model = load_int8_model(ptq_model_path)

    # QAT
    qat_payload = torch.load(qat_model_path, map_location='cpu')
    qat_model = load_int8_model(qat_model_path)

    # 2. Artifact sizes
    fp32_size_bytes = os.path.getsize(fp32_checkpoint_path)
    ptq_size_bytes = os.path.getsize(ptq_model_path)
    qat_size_bytes = os.path.getsize(qat_model_path)

    # 3. Parameter counts
    fp32_params = fp32_model.count_parameters()

    # 4. Latency benchmarks (CPU QNNPACK execution path)
    print("\nBenchmarking FP32 Model Latency...")
    fp32_bs1 = measure_inference_latency(fp32_model, batch_size=1)
    fp32_bs64 = measure_inference_latency(fp32_model, batch_size=64)

    print("Benchmarking INT8-PTQ Model Latency...")
    ptq_bs1 = measure_inference_latency(ptq_model, batch_size=1)
    ptq_bs64 = measure_inference_latency(ptq_model, batch_size=64)

    print("Benchmarking INT8-QAT Model Latency...")
    qat_bs1 = measure_inference_latency(qat_model, batch_size=1)
    qat_bs64 = measure_inference_latency(qat_model, batch_size=64)

    results = {
        'hardware_environment': {
            'platform': 'macOS Apple Silicon (ARM64)',
            'quantization_engine': 'qnnpack',
            'execution_device': 'cpu'
        },
        'artifact_sizes': {
            'fp32_bytes': fp32_size_bytes,
            'fp32_kb': round(fp32_size_bytes / 1024, 2),
            'ptq_bytes': ptq_size_bytes,
            'ptq_kb': round(ptq_size_bytes / 1024, 2),
            'qat_bytes': qat_size_bytes,
            'qat_kb': round(qat_size_bytes / 1024, 2),
            'ptq_compression_ratio': round(fp32_size_bytes / max(1, ptq_size_bytes), 2),
            'qat_compression_ratio': round(fp32_size_bytes / max(1, qat_size_bytes), 2)
        },
        'parameter_counts': {
            'total_parameters': fp32_params,
            'trainable_parameters': fp32_params
        },
        'quantized_operator_coverage': {
            'ptq': ptq_payload.get('quantized_operators', {}),
            'qat': qat_payload.get('quantized_operators', {})
        },
        'latency_edge_bs1': {
            'fp32': fp32_bs1,
            'ptq': ptq_bs1,
            'qat': qat_bs1,
            'median_latency_speedup_ptq': round(fp32_bs1['median_latency_ms'] / max(1e-6, ptq_bs1['median_latency_ms']), 2),
            'median_latency_speedup_qat': round(fp32_bs1['median_latency_ms'] / max(1e-6, qat_bs1['median_latency_ms']), 2),
            'mean_throughput_speedup_ptq': round(ptq_bs1['throughput_samples_per_sec'] / max(1e-6, fp32_bs1['throughput_samples_per_sec']), 2),
            'mean_throughput_speedup_qat': round(qat_bs1['throughput_samples_per_sec'] / max(1e-6, fp32_bs1['throughput_samples_per_sec']), 2),
            'speedup_ptq_over_fp32': round(fp32_bs1['median_latency_ms'] / max(1e-6, ptq_bs1['median_latency_ms']), 2),
            'speedup_qat_over_fp32': round(fp32_bs1['median_latency_ms'] / max(1e-6, qat_bs1['median_latency_ms']), 2)
        },
        'latency_batched_bs64': {
            'fp32': fp32_bs64,
            'ptq': ptq_bs64,
            'qat': qat_bs64,
            'median_latency_speedup_ptq': round(fp32_bs64['median_latency_ms'] / max(1e-6, ptq_bs64['median_latency_ms']), 2),
            'median_latency_speedup_qat': round(fp32_bs64['median_latency_ms'] / max(1e-6, qat_bs64['median_latency_ms']), 2),
            'mean_throughput_speedup_ptq': round(ptq_bs64['throughput_samples_per_sec'] / max(1e-6, fp32_bs64['throughput_samples_per_sec']), 2),
            'mean_throughput_speedup_qat': round(qat_bs64['throughput_samples_per_sec'] / max(1e-6, fp32_bs64['throughput_samples_per_sec']), 2),
            'speedup_ptq_over_fp32': round(fp32_bs64['median_latency_ms'] / max(1e-6, ptq_bs64['median_latency_ms']), 2),
            'speedup_qat_over_fp32': round(fp32_bs64['median_latency_ms'] / max(1e-6, qat_bs64['median_latency_ms']), 2)
        }
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nHardware benchmarks complete. Saved to {output_path}")
    print(f"  Artifact sizes: FP32={results['artifact_sizes']['fp32_kb']} KB, PTQ={results['artifact_sizes']['ptq_kb']} KB, QAT={results['artifact_sizes']['qat_kb']} KB")
    print(f"  BS=1 Median Latency: FP32={fp32_bs1['median_latency_ms']:.3f} ms, PTQ={ptq_bs1['median_latency_ms']:.3f} ms, QAT={qat_bs1['median_latency_ms']:.3f} ms")
    return results
