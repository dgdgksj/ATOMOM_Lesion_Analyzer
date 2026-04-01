import os
import sys
from functools import lru_cache
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app_config.paths import runtime_cache_path

os.environ.setdefault("YOLO_CONFIG_DIR", runtime_cache_path("tmp", "ultralytics"))


@lru_cache(maxsize=1)
def get_skin_lesion():
    from app_config.runtime_presets import (
        DEFAULT_EF_CONFIGS,
        DEFAULT_MRCNN_CONFIGS,
        DEFAULT_YOLO_CONFIGS,
    )
    from pipelines.lesion_pipeline import SkinLesionPipeline

    return SkinLesionPipeline(
        ef_configs=DEFAULT_EF_CONFIGS,
        yolo_configs=DEFAULT_YOLO_CONFIGS,
        mrcnn_configs=DEFAULT_MRCNN_CONFIGS,
    )


def analyze_image(image_path):
    return get_skin_lesion().inference(image_path=image_path)
