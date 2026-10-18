
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

import config
from utils.pose_estimator import QualityReport

STATE_PENDING = "PENDING"    
STATE_KNOWN = "KNOWN"        
STATE_UNKNOWN = "UNKNOWN"    
BBox = Tuple[int, int, int, int]


def iou_matrix(boxes_a: List[BBox], boxes_b: List[BBox]) -> np.ndarray:
    if not boxes_a or not boxes_b:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)
    a = np.asarray(boxes_a, dtype=np.float32)
    b = np.asarray(boxes_b, dtype=np.float32)
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return (inter / np.maximum(union, 1e-6)).astype(np.float32)


@dataclass
class Track:
    track_id: int
    bbox: BBox
    n_confirm: int = config.N_CONFIRM_FRAMES
    missed: int = 0                                 
    state: str = STATE_PENDING
    person_id: Optional[int] = None
    similarity: float = 0.0                          
    votes: Deque[Optional[int]] = field(default_factory=deque)
    last_quality: Optional[QualityReport] = None

    def __post_init__(self) -> None:
        self.votes = deque(maxlen=self.n_confirm)

    def vote_run(self) -> int:
        if not self.votes:
            return 0
        last, run = self.votes[-1], 0
        for v in reversed(self.votes):
            if v != last:
                break
            run += 1
        return run

    def add_vote(self, person_id: Optional[int], similarity: float) -> bool:
        self.votes.append(person_id)
        if person_id is not None and person_id == self.person_id:
            self.similarity = similarity if self.similarity <= 0 else 0.7 * self.similarity + 0.3 * similarity

        if len(self.votes) < self.n_confirm or len(set(self.votes)) != 1:
            return False

        unanimous = self.votes[0]
        new_state = STATE_UNKNOWN if unanimous is None else STATE_KNOWN
        if new_state == self.state and unanimous == self.person_id:
            return False
        self.state = new_state
        self.person_id = unanimous
        self.similarity = similarity if unanimous is not None else 0.0
        return True

    def set_identity(self, person_id: int, similarity: float = 1.0) -> None:
        self.state = STATE_KNOWN
        self.person_id = person_id
        self.similarity = similarity
        self.votes.clear()
        self.votes.extend([person_id] * self.n_confirm)


class FaceTracker:
    def __init__(
        self,
        n_confirm: int = config.N_CONFIRM_FRAMES,
        iou_threshold: float = config.TRACK_IOU_THRESHOLD,
        max_missed: int = config.TRACK_MAX_MISSED,
    ) -> None:
        self.n_confirm = n_confirm
        self.iou_threshold = iou_threshold
        self.max_missed = max_missed
        self._tracks: Dict[int, Track] = {}
        self._next_id = 1

    @property
    def tracks(self) -> Dict[int, Track]:
        return self._tracks

    def reset(self) -> None:
        self._tracks.clear()
        self._next_id = 1

    def update(self, bboxes: List[BBox]) -> List[Track]:
        track_list = list(self._tracks.values())
        assigned: List[Optional[Track]] = [None] * len(bboxes)

        if bboxes and track_list:
            iou = iou_matrix(list(bboxes), [t.bbox for t in track_list])
            pairs = [
                (float(iou[d, t]), d, t)
                for d in range(len(bboxes)) for t in range(len(track_list))
                if iou[d, t] >= self.iou_threshold
            ]
            pairs.sort(reverse=True)                 
            used_d, used_t = set(), set()
            for _, d, t in pairs:
                if d in used_d or t in used_t:
                    continue
                used_d.add(d)
                used_t.add(t)
                assigned[d] = track_list[t]

        matched_ids = set()
        for d, bbox in enumerate(bboxes):
            track = assigned[d]
            if track is None:
                track = Track(track_id=self._next_id, bbox=bbox, n_confirm=self.n_confirm)
                self._tracks[track.track_id] = track
                self._next_id += 1
                assigned[d] = track
            track.bbox = bbox
            track.missed = 0
            matched_ids.add(track.track_id)

        for tid in list(self._tracks):
            if tid not in matched_ids:
                self._tracks[tid].missed += 1
                if self._tracks[tid].missed > self.max_missed:
                    del self._tracks[tid]

        return [t for t in assigned if t is not None]
