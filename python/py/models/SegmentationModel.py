import os

import cv2
import numpy as np
import onnxruntime as ort
import torch
import torchvision.models as models


class SegmentationModel:
    def __init__(self) -> None:
        self.image_width = 640
        self.image_height = 640
        self.onnx_root = "./../onnx_models"
        self.onnx_name = "deeplabv3_mobilenet_v3_large_voc.onnx"
        self.person_label = 15

    def setup_model(self, backend="pytorch"):
        if backend == "pytorch":
            return models.segmentation.deeplabv3_mobilenet_v3_large(
                pretrained=True
            ).eval()
        elif backend == "opencv":
            onnx_path = os.path.join(self.onnx_root, self.onnx_name)
            return cv2.dnn.readNetFromONNX(onnx_path)
        elif backend == "onnx":
            onnx_path = os.path.join(self.onnx_root, self.onnx_name)
            return ort.InferenceSession(onnx_path)

    def preprocess(self, image, backend="pytorch"):
        if backend == "pytorch":
            image = cv2.resize(image, (self.image_width, self.image_height))
            tensor = torch.as_tensor(image, dtype=torch.float) / 255.0
            tensor = tensor.permute(2, 0, 1).unsqueeze(0)
            return tensor

        elif backend == "opencv" or backend == "onnx":
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
        elif backend == "opencv":
            model.setInput(tensor)
            return model.forward()
        elif backend == "onnx":
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
                color[is_person], (width, height), cv2.INTER_NEAREST
            )
            visualize = cv2.addWeighted(image, 0.5, color_result, 0.5, 1.0)
            return visualize
        elif backend == "onnx":
            index = np.squeeze(output).argmax(0)
            is_person = np.where(index == self.person_label, 1, 0).astype(int)
            color = np.array([[0, 0, 0], [0, 0, 255]], dtype=np.uint8)
            color_result = cv2.resize(
                color[is_person], (width, height), cv2.INTER_NEAREST
            )
            visualize = cv2.addWeighted(image, 0.5, color_result, 0.5, 1.0)
            return visualize
