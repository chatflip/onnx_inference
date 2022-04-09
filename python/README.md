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
| Ubuntu 18.04 | pytorch       | 111.39         | 1.00    |
|              | onnx-tensorrt | 15.86          | 7.02    |
|              | onnx-gpu      | 24.58          | 4.53    |
|              | onnx-cpu      | 65.96          | 1.69    |
| Windows 10   | pytorch       |                |         |
|              | onnx-tensorrt |                |         |
|              | onnx-gpu      |                |         |
|              | onnx-cpu      |                |         |
