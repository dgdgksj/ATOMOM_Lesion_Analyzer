from vendor.mrcnn.config import Config as MaskRCNNBaseConfig

from app_config.paths import repo_path


class EfficientNetConfig:
    weight = None
    device = None
    verbose = None
    topk = None
    model_path = None
    model_name = None
    class_names = None


class PrimaryEfficientNetConfig(EfficientNetConfig):
    weight = 0.5
    device = 0
    verbose = False
    topk = 5
    model_path = repo_path(
        "models",
        "weights",
        "classification",
        "primary_efficientnet.pt",
    )
    model_name = "efficientnet-b0"
    class_names = {
        "000": "normal_skin",
        "001": "atopy",
        "002": "prurigo",
        "003": "scar",
        "004": "psoriasis",
        "005": "varicella",
        "006": "nummular_eczema",
        "007": "ota_like_melanosis",
        "008": "becker_nevus",
        "009": "pyogenic_granuloma",
        "010": "acne",
        "011": "salmon_patches",
        "012": "dermatophytosis",
        "013": "wart",
        "014": "impetigo",
        "015": "vitiligo",
        "016": "ingrowing_nails",
        "017": "congenital_melanocytic_nevus",
        "018": "keloid",
        "019": "epidermal_cyst",
        "020": "insect_bite",
        "021": "molluscum_contagiosum",
        "022": "pityriasis_versicolor",
        "023": "melanonychia",
        "024": "alopecia_areata",
        "025": "epidermal_nevus",
        "026": "herpes_simplex",
        "027": "urticaria",
        "028": "nevus_depigmentosus",
        "029": "lichen_striatus",
        "030": "mongolian_spot_and_ectopic_mongolian_spot",
        "031": "capillary_malformation",
        "032": "pityriasis_lichenoides_chronica",
        "033": "infantile_hemangioma",
        "034": "mastocytoma",
        "035": "nevus_sebaceous",
        "036": "onychomycosis",
        "037": "milk_coffee_nevus",
        "038": "nail_dystrophy",
        "039": "melanocytic_nevus",
        "040": "juvenile_xanthogranuloma",
    }


class SecondaryEfficientNetConfig(EfficientNetConfig):
    weight = 0.5
    device = 0
    verbose = False
    topk = 5
    model_path = repo_path(
        "models",
        "weights",
        "classification",
        "secondary_efficientnet.pt",
    )
    model_name = "efficientnet-b0"
    class_names = {
        "000": "normal_skin",
        "001": "atopy",
        "002": "psoriasis",
        "003": "urticaria",
    }


class LesionYoloConfig:
    model_path = repo_path("models", "weights", "segmentation", "yolo", "lesion_yolo.pt")
    model_names = None
    display = False
    save_path = None
    verbose = False
    device = 0
    label = False
    bbox = False
    segmentation = True
    file_paths = None


class LesionMaskRCNNConfig(MaskRCNNBaseConfig):
    ROOT_DIR = repo_path()
    COCO_WEIGHTS_PATH = repo_path("mask_rcnn_coco.h5")
    DEFAULT_LOGS_DIR = repo_path("logs")
    class_names = ["others", "atopic_dermatitis", "seborrheic dermatitis", "psoriasis", "rosacea", "acne"]
    weights = repo_path("models", "weights", "segmentation", "mrcnn", "lesion_mask_rcnn.h5")
    NAME = "atopy"
    logs = DEFAULT_LOGS_DIR
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 1
    VALIDATION_STEPS = 50
    STEPS_PER_EPOCH = 1000
