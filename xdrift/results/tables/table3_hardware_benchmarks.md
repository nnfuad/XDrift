# Table 3: Hardware, Latency, and Throughput Benchmarks (macOS ARM64 QNNPACK)

*Hardware: Apple Silicon Mac (M-series, 8 cores). Backend: CPU native execution via `torch.backends.quantized.engine = 'qnnpack'`. Measurements: 50 warmup runs, 500 timed repetitions.*

### Panel A: Storage & Model Footprint
| Model Architecture | Parameter Count | Disk Checkpoint Size | Storage Reduction | Execution Path |
| :--- | :---: | :---: | :---: | :---: |
| **FP32 Baseline** | 44,129 | 184.24 KB | 1.00× (Ref) | Standard FP32 Float Operations |
| **INT8-PTQ** | 44,129 | 51.77 KB | **3.56×** | Deployed `torch.qint8` Integer Operators |
| **INT8-QAT** | 44,129 | 51.77 KB | **3.56×** | Deployed `torch.qint8` Integer Operators |

---

### Panel B: Single-Flow Streaming Latency (Edge Scenario, Batch Size = 1)
*Characterizes immediate packet-arrival triage without buffering delay.*

| Model Architecture | Median Latency | Mean Latency | Latency Std Dev | Inter-Quartile Range (IQR) | 95th Percentile (p95) | 99th Percentile (p99) | Edge Throughput | Relative Speedup |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **FP32 Baseline** | **0.258 ms** | 0.261 ms | 0.020 ms | 0.011 ms | 0.291 ms | 0.362 ms | 3,835 flows/s | 1.00× (Ref) |
| **INT8-PTQ** | 0.308 ms | 0.464 ms | 0.545 ms | 0.133 ms | 1.163 ms | 1.957 ms | 2,155 flows/s | **0.84× (Slower)** |
| **INT8-QAT** | 0.329 ms | 0.406 ms | 0.231 ms | 0.097 ms | 0.902 ms | 1.495 ms | 2,462 flows/s | **0.78× (Slower)** |

*Observation: At Batch Size = 1, quantization yields no speedup because PyTorch QNNPACK invocation, dynamic quantization of inputs, and scalar dequantization dominate computation.*

---

### Panel C: Batched Server Throughput (High-Volume Pipeline, Batch Size = 64)
*Characterizes offline audit, log ingestion, or buffered middlebox inspection.*

| Model Architecture | Median Batch Latency | Mean Batch Latency | Batch Latency Std Dev | 95th Percentile (p95) | Measured Throughput | Throughput Speedup |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **FP32 Baseline** | 9.458 ms | 13.831 ms | 38.221 ms | 20.864 ms | 4,627 flows/s | 1.00× (Ref) |
| **INT8-PTQ** | 1.458 ms | 2.097 ms | 3.825 ms | 4.175 ms | 30,518 flows/s | **6.49×** |
| **INT8-QAT** | **1.159 ms** | **1.398 ms** | **0.510 ms** | **2.408 ms** | **45,768 flows/s** | **8.16×** |

*Observation: Under batch size 64, kernel invocation overhead is amortized, allowing ARM64 NEON integer dot-product SIMD instructions to deliver a 8.16× throughput speedup.*
