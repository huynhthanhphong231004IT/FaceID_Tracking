from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

import config
from models import ModelLoadError

logger = logging.getLogger(__name__)


class ArcFaceExtractor:

    def __init__(
        self,
        model_name: str = config.INSIGHTFACE_MODEL,
        root: Path = config.INSIGHTFACE_ROOT,
        force_cpu: bool = False,
    ) -> None:
        self.model_name = model_name
        self.root = Path(root)
        self.force_cpu = force_cpu
        self.embedding_dim = config.EMBEDDING_DIM
        self._rec = None            
        self._det = None            
        self._face_align = None     
    def load(self) -> None:
        try:
            import onnxruntime as ort
            from insightface.app import FaceAnalysis
            from insightface.utils import face_align
        except ImportError as exc:
            raise ModelLoadError(
                "Thiếu thư viện InsightFace/onnxruntime. Chạy: pip install insightface onnxruntime\n"
                "(Windows: nếu lỗi biên dịch, cài 'Microsoft C++ Build Tools' rồi thử lại.)"
            ) from exc

        available = ort.get_available_providers()
        use_gpu = (not self.force_cpu) and "CUDAExecutionProvider" in available
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if use_gpu else ["CPUExecutionProvider"]
        ctx_id = 0 if use_gpu else -1

        self.root.mkdir(parents=True, exist_ok=True)
        try:
            app = FaceAnalysis(
                name=self.model_name,
                root=str(self.root),
                allowed_modules=["detection", "recognition"],
                providers=providers,
            )
            app.prepare(
                ctx_id=ctx_id,
                det_thresh=config.LANDMARK_DET_THRESH,
                det_size=config.LANDMARK_FALLBACK_SIZE,
            )
        except Exception as exc:  
            raise ModelLoadError(
                f"Không tải được model InsightFace '{self.model_name}': {exc}\n"
                f"Lần chạy đầu cần Internet để tải model vào:\n  {self.root}"
            ) from exc

        self._rec = app.models.get("recognition")
        self._det = app.models.get("detection")
        self._face_align = face_align
        if self._rec is None:
            raise ModelLoadError("Gói InsightFace không chứa model 'recognition' (ArcFace).")
        self._rec.get_feat([np.zeros((config.ALIGN_SIZE, config.ALIGN_SIZE, 3), dtype=np.uint8)])
        logger.info("ArcFace sẵn sàng (providers=%s)", providers)
    def align(self, frame: np.ndarray, kps: np.ndarray) -> np.ndarray:
        if self._face_align is None:
            raise RuntimeError("ArcFaceExtractor chưa được load().")
        return self._face_align.norm_crop(frame, np.asarray(kps, dtype=np.float32),
                                          image_size=config.ALIGN_SIZE)

    def embed(self, aligned_faces: Sequence[np.ndarray]) -> np.ndarray:

        if self._rec is None:
            raise RuntimeError("ArcFaceExtractor chưa được load().")
        n = len(aligned_faces)
        if n == 0:
            return np.zeros((0, self.embedding_dim), dtype=np.float32)

        try:
            feats = self._rec.get_feat(list(aligned_faces))
        except Exception: 
            logger.debug("Batch inference lỗi, chuyển sang chạy từng ảnh", exc_info=True)
            feats = np.concatenate([self._rec.get_feat([img]) for img in aligned_faces], axis=0)

        feats = np.asarray(feats, dtype=np.float32).reshape(n, -1)
        norms = np.linalg.norm(feats, axis=1, keepdims=True)
        return feats / np.maximum(norms, 1e-12)
    def landmarks_from_crop(
        self, frame: np.ndarray, bbox: Tuple[int, int, int, int]
    ) -> Optional[np.ndarray]:
        if self._det is None:
            return None
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        mx = int((x2 - x1) * config.LANDMARK_CROP_MARGIN)
        my = int((y2 - y1) * config.LANDMARK_CROP_MARGIN)
        cx1, cy1 = max(0, x1 - mx), max(0, y1 - my)
        cx2, cy2 = min(w, x2 + mx), min(h, y2 + my)
        if cx2 - cx1 < 24 or cy2 - cy1 < 24:
            return None

        crop = frame[cy1:cy2, cx1:cx2]
        dets, kpss = self._det.detect(crop, input_size=config.LANDMARK_FALLBACK_SIZE,
                                      max_num=1, metric="default")
        if kpss is None or len(kpss) == 0:
            return None
        kps = np.asarray(kpss[0], dtype=np.float32).copy()
        kps[:, 0] += cx1
        kps[:, 1] += cy1
        return kps
