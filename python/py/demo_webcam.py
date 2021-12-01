import argparse
import sys
import time

import cv2
import models


def conf():
    parser = argparse.ArgumentParser(description="mbg")
    parser.add_argument("--model_type", type=str, default="segmentation")
    parser.add_argument("--backend", type=str, default="onnx")
    parser.add_argument("--camera_width", type=int, default=1280)
    parser.add_argument("--camera_height", type=int, default=720)
    parser.add_argument("--fps", type=float, default=60.0)
    parser.add_argument("--buffersize", type=int, default=1)
    args = parser.parse_args()
    return args


def main():
    args = conf()
    print(args)
    model_info = models.get_model(args.model_type)
    model = model_info.setup_model(args.backend)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.camera_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.camera_height)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FPS, args.fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, args.buffersize)

    while cap.isOpened():
        ret, image = cap.read()
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
        postprocess_time = time.perf_counter() - start_time

        sys.stdout.write(
            f"\rpreprocess: {1000.0 * preprocess_time:.2f}ms\t"
            f"inference: {1000.0 * inference_time:.2f}ms\t"
            f"postprocess: {1000.0 * postprocess_time:.2f}ms"
        )
        sys.stdout.flush()
        cv2.imshow("result", result)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    print()
    sys.stdout.write("\r")
    sys.stdout.flush()
    cap.release()


if __name__ == "__main__":
    main()
