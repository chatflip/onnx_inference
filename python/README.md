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
| Windows 10   | pytorch       |                |         |
|              | onnx-tensorrt |                |         |
|              | onnx-gpu      |                |         |
|              | onnx-cpu      |                |         |
