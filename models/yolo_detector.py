from __future__ import annotations

import logging
import os
import shutil
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np

import config
from models import ModelLoadError

logger = logging.getLogger(__name__)
LandmarkFallback = Callable[[np.ndarray, Tuple[int, int, int, int]], Optional[np.ndarray]]


@dataclass
class FaceDet:
    bbox: Tuple[int, int, int, int]            
    score: float                               
    kps: Optional[np.ndarray] = None          


def resolve_device(preferred: str = "auto") -> str:
    if preferred and preferred != "auto":
        return preferred
    try:
        import torch  

        if torch.cuda.is_available():
            return "cuda:0"
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return "mps"
    except Exception:
        logger.warning("Không kiểm tra được GPU qua torch, dùng CPU.", exc_info=True)
    return "cpu"


def _download_weights(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    logger.info("Đang tải trọng số YOLO-face từ %s", url)
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(request, timeout=60) as resp, open(tmp, "wb") as fh:
            shutil.copyfileobj(resp, fh, length=1 << 20)
        os.replace(tmp, dest)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
    logger.info("Đã lưu trọng số tại %s", dest)


class YoloFaceDetector:
    def __init__(
        self,
        weights_path: Path = config.YOLO_WEIGHTS_PATH,
        weights_url: str = config.YOLO_WEIGHTS_URL,
        conf: float = config.YOLO_CONF,
        iou: float = config.YOLO_IOU,
        imgsz: int = config.YOLO_IMGSZ,
        max_det: int = config.YOLO_MAX_DET,
        device: str = config.DEVICE,
        landmark_fallback: Optional[LandmarkFallback] = None,
        force_cpu: bool = False,
    ) -> None:
        self.weights_path = Path(weights_path)
        self.weights_url = weights_url
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.max_det = max_det
        self.device = "cpu" if force_cpu else device
        self.landmark_fallback = landmark_fallback
        self.model = None

    def load(self) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ModelLoadError(
                "Chưa cài 'ultralytics'. Chạy: pip install ultralytics"
            ) from exc

        if not self.weights_path.exists():
            try:
                _download_weights(self.weights_url, self.weights_path)
            except Exception as exc:  
                raise ModelLoadError(
                    f"Không tìm thấy trọng số YOLO-face tại:\n  {self.weights_path}\n"
                    f"và tự tải thất bại ({exc}).\n"
                    f"Hãy tải thủ công file yolov8n-face.pt từ:\n  {self.weights_url}\n"
                    f"rồi đặt vào thư mục models/weights/."
                ) from exc

        self.device = resolve_device(self.device)
        try:
            self.model = YOLO(str(self.weights_path))
            dummy = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
            self.model.predict(dummy, imgsz=self.imgsz, device=self.device, verbose=False)
        except Exception as exc: 
            raise ModelLoadError(f"Không khởi tạo được YOLO-face ({self.weights_path.name}): {exc}") from exc
        logger.info("YOLO-face sẵn sàng (device=%s)", self.device)

    def detect(self, frame: np.ndarray) -> List[FaceDet]:
        if self.model is None:
            raise RuntimeError("YoloFaceDetector chưa được load().")

        results = self.model.predict(
            frame, imgsz=self.imgsz, conf=self.conf, iou=self.iou, max_det=self.max_det,
            device=self.device, verbose=False,
        )
        if not results:
            return []
        result = results[0]
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return []

        xyxy = boxes.xyxy.cpu().numpy()
        scores = boxes.conf.cpu().numpy()

        kps_all: Optional[np.ndarray] = None
        keypoints = getattr(result, "keypoints", None)
        if keypoints is not None and getattr(keypoints, "xy", None) is not None:
            arr = keypoints.xy.cpu().numpy()
            if arr.ndim == 3 and arr.shape[0] == len(xyxy) and arr.shape[1] >= 5:
                kps_all = arr[:, :5, :].astype(np.float32)

        h, w = frame.shape[:2]
        detections: List[FaceDet] = []
        for i, (box, score) in enumerate(zip(xyxy, scores)):
            x1 = int(max(0, min(w - 1, round(float(box[0])))))
            y1 = int(max(0, min(h - 1, round(float(box[1])))))
            x2 = int(max(0, min(w, round(float(box[2])))))
            y2 = int(max(0, min(h, round(float(box[3])))))
            if x2 - x1 < 8 or y2 - y1 < 8:
                continue
            bbox = (x1, y1, x2, y2)

            kps: Optional[np.ndarray] = None
            if kps_all is not None and np.any(kps_all[i]):
                kps = kps_all[i]
            elif self.landmark_fallback is not None:
                try:
                    kps = self.landmark_fallback(frame, bbox)
                except Exception:  
                    logger.debug("Landmark fallback lỗi", exc_info=True)
                    kps = None
            detections.append(FaceDet(bbox=bbox, score=float(score), kps=kps))
        return detections
