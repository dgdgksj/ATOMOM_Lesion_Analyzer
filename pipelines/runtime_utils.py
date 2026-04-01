from pathlib import Path

import natsort


try:
    from setproctitle import setproctitle as _setproctitle
except ImportError:
    def _setproctitle(_title):
        return None


VALID_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def set_process_title(title):
    _setproctitle(title)


def collect_image_paths(image_path):
    path = Path(image_path)
    if path.is_file():
        return [str(path)]
    if path.is_dir():
        file_paths = [
            str(file_path)
            for file_path in natsort.natsorted(path.iterdir(), key=lambda item: item.name)
            if file_path.is_file() and file_path.suffix.lower() in VALID_IMAGE_EXTENSIONS
        ]
        if file_paths:
            return file_paths
        raise Exception("No valid image files found, please check dir")
    raise Exception("image_path is not dir or valid image, please check image_path")
