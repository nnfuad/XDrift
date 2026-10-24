# Table 3: Hardware, Latency, and Throughput Benchmarks (macOS ARM64 QNNPACK)

*Hardware: Apple Silicon Mac (M-series, 8 cores). Backend: CPU native execution via `torch.backends.quantized.engine = 'qnnpack'`. Measurements: 50 warmup runs, 500 timed repetitions.*

### Panel A: Storage & Model Footprint
| Model Architecture | Parameter Count | Disk Checkpoint Size | Storage Reduction | Execution Path |
| :--- | :---: | :---: | :---: | :---: |
| **FP32 Baseline** | 44,129 | 183.86 KB | 1.00× (Ref) | Standard FP32 Float Operations |
| **INT8-PTQ** | 44,129 | 51.77 KB | **3.55×** | Deployed `torch.qint8` Integer Operators |
| **INT8-QAT** | 44,129 | 51.77 KB | **3.55×** | Deployed `torch.qint8` Integer Operators |

---

### Panel B: Single-Flow Streaming Latency (Edge Scenario, Batch Size = 1)
*Characterizes immediate packet-arrival triage without buffering delay.*

| Model Architecture | Median Latency | Mean Latency | Latency Std Dev | Inter-Quartile Range (IQR) | 95th Percentile (p95) | 99th Percentile (p99) | Edge Throughput | Relative Speedup |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **FP32 Baseline** | **0.273 ms** | 0.282 ms | 0.026 ms | 0.025 ms | 0.333 ms | 0.364 ms | 3,549 flows/s | 1.00× (Ref) |
| **INT8-PTQ** | 0.405 ms | 0.416 ms | 0.037 ms | 0.019 ms | 0.477 ms | 0.557 ms | 2,404 flows/s | **0.68× (Slower)** |
| **INT8-QAT** | 0.288 ms | 0.285 ms | 0.015 ms | 0.011 ms | 0.307 ms | 0.333 ms | 3,504 flows/s | **0.95× (Parity)** |

*Observation: At Batch Size = 1, quantization yields no speedup because PyTorch QNNPACK invocation, dynamic quantization of inputs, and scalar dequantization dominate computation.*

---

### Panel C: Batched Server Throughput (High-Volume Pipeline, Batch Size = 64)
*Characterizes offline audit, log ingestion, or buffered middlebox inspection.*

| Model Architecture | Median Batch Latency | Mean Batch Latency | Batch Latency Std Dev | 95th Percentile (p95) | Measured Throughput | Throughput Speedup |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **FP32 Baseline** | 13.169 ms | 13.384 ms | 1.762 ms | 16.104 ms | 4,782 flows/s | 1.00× (Ref) |
| **INT8-PTQ** | 0.932 ms | 0.975 ms | 0.129 ms | 1.292 ms | 65,650 flows/s | **13.73×** |
| **INT8-QAT** | **0.873 ms** | **0.883 ms** | **0.101 ms** | **1.080 ms** | **72,456 flows/s** | **15.08×** |

*Observation: Under batch size 64, kernel invocation overhead is amortized, allowing ARM64 NEON integer dot-product SIMD instructions to deliver a 15.08× throughput speedup.*
