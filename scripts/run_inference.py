import argparse
import json
import sys
from pathlib import Path

import cv2


REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DEFAULT_IMAGE_PATH = REPO_ROOT / "test_data" / "images" / "samples" / "normal_002.JPG"


def parse_args():
    parser = argparse.ArgumentParser(description="Run ATOMOM lesion inference from one entry point.")
    parser.add_argument(
        "mode",
        choices=["pipeline", "classifier", "yolo", "mrcnn"],
        help="Inference mode to run.",
    )
    parser.add_argument(
        "--image",
        default=str(DEFAULT_IMAGE_PATH),
        help="Image path to analyze. Defaults to the repository smoke-test sample.",
    )
    parser.add_argument(
        "--classifier",
        choices=["primary", "secondary", "all"],
        default="all",
        help="Classifier preset to use when mode=classifier.",
    )
    parser.add_argument(
        "--output",
        help="Optional path to save the annotated result image.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show visualization windows for segmentation modes.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON instead of a compact text summary.",
    )
    return parser.parse_args()


def _save_image(image, output_path):
    if not output_path:
        return
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), image)


def _print(data, as_json):
    if as_json:
        print(json.dumps(data, indent=2, ensure_ascii=True))
        return

    for key, value in data.items():
        print(f"{key}: {value}")


def run_pipeline(image_path, output_path, as_json):
    from backend.atomom.predictor import analyze_image

    result_image, predictions = analyze_image(image_path)
    _save_image(result_image, output_path)
    payload = {
        "mode": "pipeline",
        "image": image_path,
        "top_predictions": predictions,
        "saved_output": output_path or "",
    }
    _print(payload, as_json)


def _classifier_configs(selection):
    from app_config.model_registry import PrimaryEfficientNetConfig, SecondaryEfficientNetConfig

    config_map = {
        "primary": [PrimaryEfficientNetConfig],
        "secondary": [SecondaryEfficientNetConfig],
        "all": [PrimaryEfficientNetConfig, SecondaryEfficientNetConfig],
    }
    return config_map[selection]


def run_classifier(image_path, selection, as_json):
    from pipelines.efficientnet_classifier import EfficientNetClassifier

    payload = {
        "mode": "classifier",
        "image": image_path,
        "classifiers": {},
    }

    for config in _classifier_configs(selection):
        model = EfficientNetClassifier(Config=config, device=config.device)
        payload["classifiers"][config.__name__] = model.inference(image_path)

    _print(payload, as_json)


def run_yolo(image_path, output_path, show, as_json):
    from app_config.model_registry import LesionYoloConfig
    from pipelines.yolo_segmenter import YoloSegmenter

    class RuntimeYoloConfig(LesionYoloConfig):
        display = show
        file_paths = None

    model = YoloSegmenter(config=RuntimeYoloConfig)
    result_image, cropped_images, status, confidences = model.inference(
        image_info=image_path,
        visualize=show,
    )[0]
    _save_image(result_image, output_path)
    payload = {
        "mode": "yolo",
        "image": image_path,
        "detected": bool(status),
        "num_crops": len(cropped_images),
        "confidences": list(confidences),
        "saved_output": output_path or "",
    }
    _print(payload, as_json)


def run_mrcnn(image_path, output_path, show, as_json):
    from app_config.model_registry import LesionMaskRCNNConfig
    from pipelines.mask_rcnn_segmenter import MaskRCNNSegmenter

    model = MaskRCNNSegmenter(mrcnn_config=LesionMaskRCNNConfig())
    result_image, cropped_images, status, confidences = model.inference(
        image_info=image_path,
        display=show,
        show_label=show,
        show_bbox=False,
    )[0]
    _save_image(cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR), output_path)
    payload = {
        "mode": "mrcnn",
        "image": image_path,
        "detected": bool(status),
        "num_crops": len(cropped_images),
        "confidences": list(confidences),
        "saved_output": output_path or "",
    }
    _print(payload, as_json)


def main():
    args = parse_args()
    image_path = str(Path(args.image).resolve())

    if args.mode == "pipeline":
        run_pipeline(image_path, args.output, args.json)
        return
    if args.mode == "classifier":
        run_classifier(image_path, args.classifier, args.json)
        return
    if args.mode == "yolo":
        run_yolo(image_path, args.output, args.show, args.json)
        return
    run_mrcnn(image_path, args.output, args.show, args.json)


if __name__ == "__main__":
    main()
