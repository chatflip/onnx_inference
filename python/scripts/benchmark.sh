#!/bin/bash
# segmentation
#python py/convert.py

# benchmark 
python py/benchmark.py --backend pytorch
python py/benchmark.py --backend onnx-tensorrt
python py/benchmark.py --backend onnx-gpu
python py/benchmark.py --backend onnx-cpu
