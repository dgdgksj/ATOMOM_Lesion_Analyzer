from app_config.model_registry import (
    LesionYoloConfig,
    PrimaryEfficientNetConfig,
    SecondaryEfficientNetConfig,
)


DEFAULT_EF_CONFIGS = [
    PrimaryEfficientNetConfig,
    SecondaryEfficientNetConfig,
]
DEFAULT_YOLO_CONFIGS = [LesionYoloConfig]
DEFAULT_MRCNN_CONFIGS = []
