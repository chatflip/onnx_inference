import argparse
import os
import time

import cv2
import models


def conf():
    parser = argparse.ArgumentParser(description="inference")
    parser.add_argument("--model_type", type=str, default="segmentation")
    parser.add_argument("--num_iteration", type=int, default=100)
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
    tensor = model_info.preprocess(image, args.backend)
    start_time = time.perf_counter()
    for _ in range(args.num_iteration):
        tensor = model_info.preprocess(image, args.backend)
    preprocess_time = time.perf_counter() - start_time
    preprocess_time /= args.num_iteration

    # inference
    output = model_info.inference(model, tensor, args.backend)
    start_time = time.perf_counter()
    for _ in range(args.num_iteration):
        output = model_info.inference(model, tensor, args.backend)
    inference_time = time.perf_counter() - start_time
    inference_time /= args.num_iteration

    # postprocess
    result = model_info.postprocess(image, output, args.backend)
    start_time = time.perf_counter()
    for _ in range(args.num_iteration):
        result = model_info.postprocess(image, output, args.backend)
    postprocess_time = time.perf_counter() - start_time
    postprocess_time /= args.num_iteration

    print(
        f"preprocess: {1000.0 * preprocess_time:.2f}ms\t"
        f"inference: {1000.0 * inference_time:.2f}ms\t"
        f"postprocess: {1000.0 * postprocess_time:.2f}ms\t"
    )


if __name__ == "__main__":
    main()
