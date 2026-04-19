# TinyML ECG Arrhythmia Detection for Low-Resource Settings

This repository implements a TinyML-ready ECG arrhythmia detection pipeline designed for commodity CPU deployment in low-resource African health settings.

## Overview

The project explores a lightweight 1D convolutional neural network for binary ECG classification using the MIT-BIH Arrhythmia Database. The goal is to detect abnormal heartbeats while preserving high accuracy and enabling efficient on-device inference through TensorFlow Lite quantization.

## Contents

- `main.ipynb` — Primary notebook containing the full experimental workflow:
  - data loading and preprocessing
  - heartbeat normalization and class balancing
  - model definition and training
  - evaluation and metrics
  - TensorFlow Lite conversion and quantization
  - CPU inference benchmarking
- `README.md` — This project summary and usage guide.

## Key Features

- Binary classification: normal vs. abnormal ECG beats
- Lightweight 1D CNN architecture optimized for embedded inference
- Patient-wise dataset split matching MIT-BIH/AAMI standards
- Training pipeline with normalization and class balancing
- Conversion to TensorFlow Lite formats:
  - float32
  - float16
  - INT8
- CPU inference benchmarking for low-resource deployment

## Dataset

The notebook expects the MIT-BIH Arrhythmia Database.

> The dataset is not included in this repository.

In the notebook, data paths are configured for a Google Drive-mounted folder. Update `dataset_path` to your local dataset location before running.

## Model Approach

The notebook builds and compares two lightweight CNN models for 1D ECG waveform classification. The general workflow is:

1. Load MIT-BIH ECG records and annotations
2. Extract individual heartbeat segments from lead II
3. Normalize each beat to zero mean and unit variance
4. Balance the training set with resampling of abnormal beats
5. Train a compact Conv1D model with global average pooling
6. Evaluate on a held-out test set
7. Convert the trained model to TensorFlow Lite for TinyML use

## Quantization and Deployment

The notebook includes TensorFlow Lite conversion experiments:

- float32 model export
- INT8 quantization with representative dataset calibration
- float16 quantization for better accuracy–size tradeoff

It also benchmarks CPU inference latency and throughput for deployment evaluation.

## Example Results

Based on the notebook results, the approximate performance was:

- `Float32`: ~95.9% accuracy
- `Float16`: ~95.9% accuracy with reduced model size (~24 KB)
- `INT8`: lower accuracy in early experiments, showing the need for robust calibration

> Exact numbers may vary by dataset split and environment.

## Requirements

Install the required Python packages before running the notebook:

```bash
pip install wfdb tensorflow numpy matplotlib scikit-learn
```

## Running the Notebook

Open `main.ipynb` in Jupyter or Google Colab, then:

1. Update the dataset path
2. Run the cells sequentially
3. Train the model and evaluate performance
4. Convert to TensorFlow Lite and benchmark inference

## Notes

- This repository is a research prototype, not a production-ready medical device.
- Use the MIT-BIH dataset responsibly and comply with applicable ethics and privacy guidelines.
- For deployment on low-resource devices, prefer the `float16` TFLite model variant when accuracy must be balanced with size.

## License

Use this repository for research and experimentation. Adapt the README or code as needed for your own paper and deployment pipeline.
