from pathlib import Path

import cv2
from django.core.files.storage import FileSystemStorage
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt

from .predictor import analyze_image


TEMPLATE_NAME = "lesion_analyzer.html"
STATIC_UPLOAD_DIR = Path(__file__).resolve().parents[1] / "static" / "uploads"


def _default_context():
    return {
        "menutitle": "ATOMOM",
        "datas": [1, 2, 3, 4, 5, 6, 7, 8],
        "originaltext": "",
        "resulttext": "",
        "srcImgname": "",
        "resultImgname": "",
    }


def _format_predictions(soft_voting_result):
    formatted_original = []
    formatted_result = []

    for index in range(0, len(soft_voting_result), 2):
        class_name = soft_voting_result[index]
        probability = soft_voting_result[index + 1]
        formatted_original.append(f"{class_name}: {probability}")

        percentage = round(probability * 100, 4)
        if percentage >= 1:
            formatted_result.append(f"{class_name}: {percentage}%")

    return "\n".join(formatted_original), "\n".join(formatted_result)


@csrf_exempt
def analyze_lesion(request):
    context = _default_context()
    image_file = request.FILES.get("image_file")

    if request.method == "POST" and image_file:
        STATIC_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        storage = FileSystemStorage(location=str(STATIC_UPLOAD_DIR))
        saved_name = storage.save(f"input-{image_file.name}", image_file)
        image_path = Path(storage.path(saved_name))

        result_image, soft_voting_result = analyze_image(str(image_path))
        result_name = f"{image_path.stem}-annotated.jpg"
        cv2.imwrite(str(image_path.with_name(result_name)), result_image)

        context["srcImgname"] = image_path.name
        context["resultImgname"] = result_name
        context["originaltext"], context["resulttext"] = _format_predictions(soft_voting_result)

    return render(request, TEMPLATE_NAME, context)


home = analyze_lesion
