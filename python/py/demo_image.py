import argparse
import os
import time

import cv2
import models


def conf():
    parser = argparse.ArgumentParser(description="mbg")
    parser.add_argument("--model_type", type=str, default="classification")
    parser.add_argument("--backend", type=str, default="onnx")
    args = parser.parse_args()
    return args


def main():
    args = conf()
    print(args)
    model_info = models.get_model(args.model_type)
    image_path = os.path.join("./../images", "test_image.jpg")
    if not os.path.exists(image_path):
        print(f"no such file: {image_path}")
    image = cv2.imread(image_path)
    model = model_info.setup_model(args.backend)

    # preprocess
    start_time = time.perf_counter()
    tensor = model_info.preprocess(image, args.backend)
    preprocess_time = time.perf_counter() - start_time

    # inference
    start_time = time.perf_counter()
    output = model_info.inference(model, tensor, args.backend)
    inference_time = time.perf_counter() - start_time

    # postprocess
    start_time = time.perf_counter()
    result = model_info.postprocess(image, output, args.backend)
    cv2.imwrite("result.png", result)
    postprocess_time = time.perf_counter() - start_time

    print(
        f"preprocess: {1000.0 * preprocess_time:.2f}ms\t"
        f"inference: {1000.0 * inference_time:.2f}ms\t"
        f"postprocess: {1000.0 * postprocess_time:.2f}ms\t"
    )


if __name__ == "__main__":
    main()
