import os

import cv2
import numpy as np
import onnx
import onnxruntime as ort
import torch
import torchvision.models as models
from onnxsim import simplify


class SegmentationModel:
    def __init__(self) -> None:
        self.image_width = 640
        self.image_height = 640
        self.onnx_root = os.path.join(".", "..","models", "onnx")
        self.onnx_name = "deeplabv3_mobilenet_v3_large_voc.onnx"
        self.tensorrt_cache = os.path.join(".", "..","models", "tensorrt")
        self.person_label = 15
        os.makedirs(self.onnx_root, exist_ok=True)

    def setup_model(self, backend="pytorch"):
        if backend == "pytorch":
            return models.segmentation.deeplabv3_mobilenet_v3_large(
                pretrained=True
            ).eval()
        elif backend == "onnx-tensorrt":
            onnx_path = os.path.join(self.onnx_root, self.onnx_name)
            providers = [
                (
                    "TensorrtExecutionProvider",
                    {
                        "trt_engine_cache_enable": True,
                        "trt_engine_cache_path": self.tensorrt_cache,
                        "trt_fp16_enable": True,
                    },
                )
            ]
            return ort.InferenceSession(onnx_path, providers=providers)
        elif backend == "onnx-gpu":
            onnx_path = os.path.join(self.onnx_root, self.onnx_name)
            providers = [
                (
                    "CUDAExecutionProvider",
                    {
                        "device_id": 0,
                        "gpu_mem_limit": 4 * 1024 * 1024 * 1024,
                        "cudnn_conv_algo_search": "EXHAUSTIVE",
                    },
                ),
            ]
            return ort.InferenceSession(onnx_path, providers=providers)
        elif backend == "onnx-cpu":
            onnx_path = os.path.join(self.onnx_root, self.onnx_name)
            providers = [
                "CPUExecutionProvider",
            ]
            return ort.InferenceSession(onnx_path, providers=providers)

    def preprocess(self, image, backend="pytorch"):
        if backend == "pytorch":
            image = cv2.resize(image, (self.image_width, self.image_height))
            tensor = torch.as_tensor(image, dtype=torch.float) / 255.0
            tensor = tensor.permute(2, 0, 1).unsqueeze(0)
            return tensor
        elif "onnx" in backend:
            image = cv2.resize(image, (self.image_width, self.image_height))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            tensor = image.astype(np.float32) / 255.0
            tensor = np.expand_dims(tensor.transpose(2, 0, 1), 0).astype(np.float32)
            return tensor

    def inference(self, model, tensor, backend="pytorch"):
        if backend == "pytorch":
            with torch.inference_mode():
                output = model(tensor)["out"].squeeze(0)
            return output
        elif "onnx" in backend:
            input_name = model.get_inputs()[0].name
            output_name = model.get_outputs()[0].name
            return model.run([output_name], {input_name: tensor})

    def postprocess(self, image, output, backend="pytorch"):
        height, width = image.shape[:2]
        if backend == "pytorch":
            index = output.squeeze().argmax(0).numpy()
            is_person = np.where(index == self.person_label, 1, 0).astype(int)
            color = np.array([[0, 0, 0], [0, 0, 255]], dtype=np.uint8)
            color_result = cv2.resize(
                color[is_person], (width, height), interpolation=cv2.INTER_NEAREST
            )
            visualize = cv2.addWeighted(image, 0.5, color_result, 0.5, 1.0)
            return visualize
        elif "onnx" in backend:
            index = np.squeeze(output).argmax(0)
            is_person = np.where(index == self.person_label, 1, 0).astype(int)
            color = np.array([[0, 0, 0], [0, 0, 255]], dtype=np.uint8)
            color_result = cv2.resize(
                color[is_person], (width, height), interpolation=cv2.INTER_NEAREST
            )
            visualize = cv2.addWeighted(image, 0.5, color_result, 0.5, 1.0)
            return visualize

    def convert_onnx(self):
        dummy_input = torch.randn(1, 3, self.image_height, self.image_width)
        dst_path = os.path.join(self.onnx_root, self.onnx_name)
        mo = self.setup_model("pytorch")
        torch.onnx.export(
            self.setup_model("pytorch"),
            dummy_input,
            dst_path,
            verbose=False,
            export_params=True,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            opset_version=12,
        )
        model_simp, check = simplify(dst_path)
        assert check, "Simplified ONNX model could not be validated"
        onnx.save(model_simp, dst_path)
        onnx_model = onnx.load(dst_path)
        onnx.checker.check_model(onnx_model)
