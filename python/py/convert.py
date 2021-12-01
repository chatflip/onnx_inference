import argparse
import os

import models
import onnx
import torch
from onnxsim import simplify


def conf():
    parser = argparse.ArgumentParser(description="mbg")
    parser.add_argument("--model_type", type=str, default="classification")
    parser.add_argument("--backend", type=str, default="onnx")
    args = parser.parse_args()
    return args


def main():
    args = conf()
    print(args)
    model = models.get_model(args.model_type)
    os.makedirs(model.onnx_root, exist_ok=True)
    dst_path = os.path.join(model.onnx_root, model.onnx_name)
    dummy_input = torch.randn(1, 3, model.image_height, model.image_width)
    torch.onnx.export(
        model.setup_model("pytorch"),
        dummy_input,
        dst_path,
        verbose=False,
        export_params=True,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        opset_version=13,
    )
    model_simp, check = simplify(dst_path)
    assert check, "Simplified ONNX model could not be validated"
    onnx.save(model_simp, dst_path)

    onnx_model = onnx.load(dst_path)
    onnx.checker.check_model(onnx_model)


if __name__ == "__main__":
    main()
