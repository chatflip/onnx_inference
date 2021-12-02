#!/bin/bash

# classification
#python py/convert.py
# pytorch 
#python py/demo_image.py --backend pytorch
# onnx inference backend opencv
#python py/demo_image.py --backend opencv
# onnx inference backend onnx
#python py/demo_image.py --backend onnx
# openvino inference backend openvino
#python py/demo_image.py --backend openvino

# classification
#python py/convert.py --model_type classification
# pytorch 
#python py/demo_image.py --model_type classification --backend pytorch
# onnx inference backend onnx
#python py/demo_image.py --model_type classification --backend onnx
# onnx inference backend opencv
#python py/demo_image.py --model_type classification --backend opencv
# openvino inference backend openvino
#python py/demo_image.py --model_type classification --backend openvino
