# onnx_inference_python

## Installation

### Windows

```bash
# comment out torch, torchvision
poetry install
poetry run pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu113
```

### Linux

```bash
poetry install
```

## Results

| OS           | backend       | inference [ms] | SpeedUp |
| ------------ | ------------- | -------------- | ------- |
| Ubuntu 18.04 | pytorch-gpu   | 8.11           | 1.00    |
|              | pytorch-cpu   | 111.39         | 0.07    |
|              | onnx-tensorrt | 15.86          | 0.51    |
|              | onnx-gpu      | 24.58          | 0.33    |
|              | onnx-cpu      | 65.96          | 0.12    |
| Windows 10   | pytorch-gpu   | 14.56          | 0.56    |
|              | pytorch-cpu   | 251.28         | 0.03    |
|              | onnx-tensorrt | 17.64          | 0.50    |
|              | onnx-gpu      | 26.69          | 0.30    |
|              | onnx-cpu      | 72.29          | 0.11    |
