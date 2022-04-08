import argparse

import models


def conf():
    parser = argparse.ArgumentParser(description="inference")
    parser.add_argument("--model_type", type=str, default="segmentation")
    args = parser.parse_args()
    return args


def main():
    args = conf()
    print(args)
    model = models.get_model(args.model_type)
    model.convert_onnx()


if __name__ == "__main__":
    main()
