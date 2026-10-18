
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

import config
_MODEL_POINTS = np.array(
    [
        [-31.5, -35.8, 38.0],  
        [31.5, -35.8, 38.0],    
        [0.0, 0.0, 0.0],        
        [-26.1, 37.0, 22.0],    
        [26.1, 37.0, 22.0],     
    ],
    dtype=np.float64,
)


@dataclass
class QualityReport:
    yaw: Optional[float] = None
    pitch: Optional[float] = None
    roll: Optional[float] = None
    sharpness: float = 0.0
    is_good: bool = False          
    reason: str = ""               


class PoseEstimator:
    def __init__(
        self,
        max_yaw: float = config.POSE_MAX_YAW,
        max_pitch: float = config.POSE_MAX_PITCH,
        max_roll: float = config.POSE_MAX_ROLL,
        blur_threshold: float = config.BLUR_THRESHOLD,
        min_face_size: int = config.MIN_FACE_SIZE,
    ) -> None:
        self.max_yaw = max_yaw
        self.max_pitch = max_pitch
        self.max_roll = max_roll
        self.blur_threshold = blur_threshold
        self.min_face_size = min_face_size

    def estimate_pose(
        self, kps: np.ndarray, frame_shape: Tuple[int, ...]
    ) -> Optional[Tuple[float, float, float]]:
        pts = np.asarray(kps, dtype=np.float64).reshape(5, 2)
        if not np.all(np.isfinite(pts)):
            return None
        eye_dist = float(np.linalg.norm(pts[1] - pts[0]))
        if eye_dist < 4.0:
            return None

        h, w = frame_shape[:2]
        focal = float(w)  
        cx, cy = float(pts[:, 0].mean()), float(pts[:, 1].mean())
        camera = np.array([[focal, 0, cx], [0, focal, cy], [0, 0, 1]], dtype=np.float64)

        try:
            ok, rvec, tvec = cv2.solvePnP(_MODEL_POINTS, pts, camera, None, flags=cv2.SOLVEPNP_EPNP)
            if not ok:
                return None
            ok, rvec, tvec = cv2.solvePnP(
                _MODEL_POINTS, pts, camera, None, rvec, tvec, True, cv2.SOLVEPNP_ITERATIVE
            )
        except cv2.error:
            return None
        if not ok or float(np.asarray(tvec, dtype=np.float64).reshape(-1)[2]) <= 0:
            return None  
        rot, _ = cv2.Rodrigues(rvec)
        yaw = math.degrees(math.asin(max(-1.0, min(1.0, -rot[2, 0]))))
        pitch = math.degrees(math.atan2(rot[2, 1], rot[2, 2]))
        roll = math.degrees(math.atan2(rot[1, 0], rot[0, 0]))
        return yaw, pitch, roll

    @staticmethod
    def sharpness(face_bgr: np.ndarray) -> float:
        gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY) if face_bgr.ndim == 3 else face_bgr
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())
    def assess(
        self,
        bbox: Tuple[int, int, int, int],
        kps: Optional[np.ndarray],
        aligned_face: Optional[np.ndarray],
        frame_shape: Tuple[int, ...],
    ) -> QualityReport:
        report = QualityReport()
        reasons = []

        x1, y1, x2, y2 = bbox
        size = min(x2 - x1, y2 - y1)
        if size < self.min_face_size:
            reasons.append(f"Mặt nhỏ ({size}px)")

        pose = self.estimate_pose(kps, frame_shape) if kps is not None else None
        if pose is None:
            reasons.append("Không ước lượng được góc mặt")
        else:
            report.yaw, report.pitch, report.roll = pose
            if abs(pose[0]) > self.max_yaw:
                reasons.append(f"Yaw {pose[0]:+.0f}°")
            if abs(pose[1]) > self.max_pitch:
                reasons.append(f"Pitch {pose[1]:+.0f}°")
            if abs(pose[2]) > self.max_roll:
                reasons.append(f"Roll {pose[2]:+.0f}°")

        if aligned_face is not None:
            report.sharpness = self.sharpness(aligned_face)
            if report.sharpness < self.blur_threshold:
                reasons.append(f"Mờ ({report.sharpness:.0f})")

        report.is_good = not reasons
        report.reason = "; ".join(reasons)
        return report
