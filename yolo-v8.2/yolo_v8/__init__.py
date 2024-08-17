# Ultralytics YOLO 🚀, AGPL-3.0 license

__version__ = "8.2.0"

from yolo_v8.data.explorer.explorer import Explorer
from yolo_v8.models import RTDETR, SAM, YOLO, YOLOWorld
from yolo_v8.models.fastsam import FastSAM
from yolo_v8.models.nas import NAS
from yolo_v8.utils import ASSETS, SETTINGS
from yolo_v8.utils.checks import check_yolo as checks
from yolo_v8.utils.downloads import download

settings = SETTINGS
__all__ = (
    "__version__",
    "ASSETS",
    "YOLO",
    "YOLOWorld",
    "NAS",
    "SAM",
    "FastSAM",
    "RTDETR",
    "checks",
    "download",
    "settings",
    "Explorer",
)
