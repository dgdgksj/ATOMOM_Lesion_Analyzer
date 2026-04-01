import csv
import os

import cv2
import numpy as np
import torch
import torch.onnx
from PIL import Image
from torchvision import transforms

from app_config.paths import experiment_results_path, repo_path
from pipelines.runtime_utils import collect_image_paths, set_process_title
from vendor.efficientnet_pytorch import EfficientNet

get_images_paths = collect_image_paths


class EfficientNetClassifier:
    def __init__(self, Config, device="cpu"):
        self.device = self.__resolve_device(device)
        self.Config = Config
        self.Config.num_classes = len(self.Config.class_names)
        self.model = self.__load_model()
        self.data_transforms1 = torch.nn.Sequential(
            transforms.Resize((224, 224)),
        )
        self.data_transforms2 = torch.nn.Sequential(
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        )

    def __resolve_device(self, device):
        if isinstance(device, torch.device):
            return device
        if isinstance(device, int):
            if torch.cuda.is_available():
                return torch.device(f"cuda:{device}")
            return torch.device("cpu")
        if isinstance(device, str):
            if device.startswith("cuda") and not torch.cuda.is_available():
                return torch.device("cpu")
            return torch.device(device)
        return torch.device("cpu")

    def __load_model(self):
        model = EfficientNet.from_name(self.Config.model_name, num_classes=self.Config.num_classes)
        model.load_state_dict(torch.load(self.Config.model_path, map_location=self.device))
        model.eval()
        return model.to(self.device)

    def __load_image(self, image_info):
        if isinstance(image_info, str):
            return Image.open(image_info).convert("RGB")
        if isinstance(image_info, np.ndarray):
            if image_info.ndim == 2:
                return Image.fromarray(image_info).convert("RGB")
            image = cv2.cvtColor(image_info, cv2.COLOR_BGR2RGB)
            return Image.fromarray(image)
        raise TypeError("Expected an image path or ndarray input")

    def inference(self, image_info, verbose=False):
        image = self.__load_image(image_info)

        to_tensor = transforms.ToTensor()
        inputs = self.data_transforms1(image)
        inputs = to_tensor(inputs).to(self.device)
        inputs = self.data_transforms2(inputs)
        outputs = self.model(inputs.unsqueeze(0))
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        top_p, top_class = probabilities.topk(self.Config.num_classes, dim=1)
        result = list(
            zip([t.item() for t in top_p.squeeze().squeeze()], [t.item() for t in top_class.squeeze().squeeze()]))
        result.sort(key=lambda x: x[1])
        classification_result = [None for _ in range(self.Config.num_classes)]
        for i, data in enumerate(result):
            class_prob, class_id = data
            class_name = self.Config.class_names[str(class_id).zfill(3)]
            classification_result[i] = (class_name, class_prob)
        classification_result.sort(key=lambda x: x[1], reverse=True)
        if self.Config.verbose or verbose:
            print(classification_result)
        return classification_result[:self.Config.topk]


class Skin_lesion:
    def __init__(self, Config_41=None):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.Config_41 = Config_41
        self.ef_41 = self.__load_ef_net(Config=self.Config_41, device=self.device)

    def __load_ef_net(self, Config, device):
        return EfficientNetClassifier(Config=Config, device=device)

    def inference(self, image_path):
        class_name, confidence = self.ef_41.inference(image_path)[0]
        file_name = os.path.basename(image_path).split(".")[0]
        print(file_name, class_name, confidence)
        return file_name, class_name, confidence


if __name__ == "__main__":
    set_process_title("lesion")

    class Config_41:
        verbose = False
        topk = 3
        model_path = repo_path("models", "weights", "classification", "primary_efficientnet.pt")
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

    image_path = repo_path("test_data", "images", "samples")
    skin_lesion = Skin_lesion(Config_41=Config_41)

    image_path_list = collect_image_paths(image_path)
    output = []
    for image_path in image_path_list:
        file_name, class_name, confidence = skin_lesion.inference(image_path=image_path)
        output.append([file_name, class_name, confidence])

    output_path = experiment_results_path("classifier_demo_output.csv")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(output)


Efficient_net = EfficientNetClassifier
