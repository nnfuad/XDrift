# Table 3: Hardware and Inference Benchmarks (macOS ARM64 QNNPACK)

| Model Condition | Artifact Size (KB) | Compression Ratio | Edge BS=1 Latency (Median ms) | BS=1 IQR (ms) | BS=1 p95 (ms) | Throughput (samples/s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **FP32 Baseline** | 184.24 KB | 1.00x | 0.267 ms | 0.042 ms | 0.405 ms | 3438.7 |
| **FP32-FT (Control)** | 184.24 KB | 1.00x | ~0.267 ms | ~0.012 ms | ~0.310 ms | ~3,439 |
| **INT8-PTQ** | 51.77 KB | 3.56x | 0.255 ms | 0.032 ms | 0.309 ms | 3796.6 |
| **INT8-QAT** | 51.77 KB | 3.56x | 0.242 ms | 0.012 ms | 0.281 ms | 4005.3 |
