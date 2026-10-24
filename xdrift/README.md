# XDrift: Investigating Attribution Drift in INT8-Quantized 1D CNNs for Encrypted Traffic Classification

## Research Objective
When a 1D convolutional neural network (1D-CNN) is quantized from FP32 to INT8 for encrypted traffic classification, does the quantization process change its feature-attribution explanations, even when predictive performance remains approximately unchanged?

The study distinguishes three outcomes:
1. **Predictive drift**: Changes in model predictions and predictive performance caused by quantization.
2. **Attribution drift**: Changes in the explanations generated for FP32 and INT8 models on the same inputs.
3. **Operational implications**: Whether measured attribution changes could affect security decisions (evaluated as formal hypotheses, not assumed).

## Repository Structure
```
xdrift/
├── README.md
├── requirements.txt
├── configs/
│   ├── data.yaml
│   ├── model.yaml
│   ├── training.yaml
│   ├── quantization.yaml
│   ├── explainability.yaml
│   └── evaluation.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   └── manifests/
├── src/
│   ├── data/
│   ├── features/
│   ├── models/
│   ├── training/
│   ├── quantization/
│   ├── explainability/
│   ├── evaluation/
│   ├── statistics/
│   └── utils/
├── tests/
├── scripts/
├── artifacts/
│   ├── checkpoints/
│   ├── quantized_models/
│   └── manifests/
├── results/
│   ├── predictions/
│   ├── attributions/
│   ├── metrics/
│   ├── figures/
│   └── tables/
└── reports/
```
