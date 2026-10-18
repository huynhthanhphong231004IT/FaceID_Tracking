from __future__ import annotations

import csv
import json
import logging
import os
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

import config

logger = logging.getLogger(__name__)

CSV_HEADER = ["ID", "Name", "Feature_Vector_JSON", "Updated_At", "Num_Features"]


def l2_normalize(vec: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    vec = np.asarray(vec, dtype=np.float32)
    if vec.ndim == 1:
        return vec / max(float(np.linalg.norm(vec)), eps)
    return vec / np.maximum(np.linalg.norm(vec, axis=1, keepdims=True), eps)


def _now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@dataclass
class PersonRecord:
    person_id: int
    name: str
    sum_vec: np.ndarray            
    mean: np.ndarray               
    count: int                     
    updated_at: str
    last_update_ts: float = 0.0    


class FeatureManager:
    def __init__(
        self,
        csv_path: Path = config.CSV_PATH,
        dim: int = config.EMBEDDING_DIM,
        max_k: int = config.MAX_FEATURES_PER_PERSON,
        autosave_interval: float = config.DB_AUTOSAVE_INTERVAL,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.dim = dim
        self.max_k = max(1, max_k)
        self.autosave_interval = autosave_interval

        self._lock = threading.RLock()
        self._persons: Dict[int, PersonRecord] = {}
        self._ids: List[int] = []                       
        self._matrix = np.zeros((0, dim), dtype=np.float32)
        self._dirty = False
        self._last_save = time.time()
    def load(self) -> int:
        with self._lock:
            self._persons.clear()
            if not self.csv_path.exists():
                self.csv_path.parent.mkdir(parents=True, exist_ok=True)
                self._write_csv()
                self._rebuild_matrix()
                return 0

            try:
                with open(self.csv_path, "r", encoding="utf-8-sig", newline="") as fh:
                    reader = csv.DictReader(fh)
                    for line_no, row in enumerate(reader, start=2):
                        try:
                            self._ingest_row(row)
                        except Exception as exc:
                            logger.warning("Bỏ qua dòng %d trong %s: %s", line_no, self.csv_path.name, exc)
            except (OSError, csv.Error) as exc:
                logger.error("Không đọc được %s: %s", self.csv_path, exc)

            self._rebuild_matrix()
            logger.info("Đã nạp %d người từ %s", len(self._persons), self.csv_path.name)
            return len(self._persons)

    def _ingest_row(self, row: Dict[str, str]) -> None:
        person_id = int(str(row["ID"]).strip())
        name = str(row.get("Name", "")).strip() or f"Person_{person_id}"
        vec = np.asarray(json.loads(row["Feature_Vector_JSON"]), dtype=np.float32).reshape(-1)
        if vec.shape[0] != self.dim:
            raise ValueError(f"vector có {vec.shape[0]} chiều, cần {self.dim}")
        if not np.all(np.isfinite(vec)) or float(np.linalg.norm(vec)) < 1e-6:
            raise ValueError("vector không hợp lệ")
        try:
            count = max(1, int(str(row.get("Num_Features", "1")).strip() or 1))
        except ValueError:
            count = 1
        mean = l2_normalize(vec)
        self._persons[person_id] = PersonRecord(
            person_id=person_id,
            name=name,
            sum_vec=mean * float(min(count, self.max_k)),
            mean=mean,
            count=count,
            updated_at=str(row.get("Updated_At", "")) or _now_str(),
        )

    def save(self, force: bool = False) -> bool:
        with self._lock:
            if not (force or self._dirty):
                return False
            ok = self._write_csv()
            if ok:
                self._dirty = False
                self._last_save = time.time()
            return ok

    def autosave(self) -> None:
        with self._lock:
            if self._dirty and (time.time() - self._last_save) >= self.autosave_interval:
                self.save()

    def _write_csv(self) -> bool:
        tmp = self.csv_path.with_suffix(".csv.tmp")
        try:
            self.csv_path.parent.mkdir(parents=True, exist_ok=True)
            with open(tmp, "w", encoding="utf-8", newline="") as fh:
                writer = csv.writer(fh)
                writer.writerow(CSV_HEADER)
                for pid in sorted(self._persons):
                    rec = self._persons[pid]
                    vec_json = json.dumps([round(float(x), 6) for x in rec.mean])
                    writer.writerow([rec.person_id, rec.name, vec_json, rec.updated_at, rec.count])
            os.replace(tmp, self.csv_path)
            return True
        except OSError as exc:
            logger.error("Không ghi được %s: %s", self.csv_path, exc)
            return False
    def _rebuild_matrix(self) -> None:
        self._ids = sorted(self._persons)
        if self._ids:
            self._matrix = np.stack([self._persons[i].mean for i in self._ids]).astype(np.float32)
        else:
            self._matrix = np.zeros((0, self.dim), dtype=np.float32)

    def num_persons(self) -> int:
        with self._lock:
            return len(self._persons)

    def get_person(self, person_id: Optional[int]) -> Optional[PersonRecord]:
        with self._lock:
            return self._persons.get(person_id) if person_id is not None else None

    def get_name(self, person_id: Optional[int]) -> str:
        rec = self.get_person(person_id)
        return rec.name if rec else "Unknown"

    def get_count(self, person_id: Optional[int]) -> int:
        rec = self.get_person(person_id)
        return rec.count if rec else 0

    def list_persons(self) -> List[Tuple[int, str, int]]:
        with self._lock:
            return [(i, self._persons[i].name, self._persons[i].count) for i in sorted(self._persons)]

    def distance_matrix(self, embeddings: np.ndarray) -> Tuple[np.ndarray, List[int]]:
        embs = np.atleast_2d(np.asarray(embeddings, dtype=np.float32))
        with self._lock:
            if not self._ids:
                return np.zeros((embs.shape[0], 0), dtype=np.float32), []
            sims = embs @ self._matrix.T
            return (1.0 - sims).astype(np.float32), list(self._ids)

    def _next_id(self) -> int:
        return (max(self._persons) + 1) if self._persons else 1

    @staticmethod
    def is_auto_name(name: str) -> bool:
        return str(name).startswith(config.AUTO_NAME_PREFIX)

    def add_person(self, name: Optional[str], embeddings: Sequence[np.ndarray]) -> PersonRecord:
        embs = np.atleast_2d(np.asarray(embeddings, dtype=np.float32))
        if embs.shape[0] == 0 or embs.shape[1] != self.dim:
            raise ValueError("Danh sách embedding rỗng hoặc sai số chiều.")
        with self._lock:
            pid = self._next_id()
            name = " ".join(str(name).split()) if name else ""
            name = name or f"{config.AUTO_NAME_PREFIX}{pid:05d}"
            total = embs.sum(axis=0)
            rec = PersonRecord(
                person_id=pid, name=name, sum_vec=total.astype(np.float32),
                mean=l2_normalize(total), count=int(embs.shape[0]), updated_at=_now_str(),
                last_update_ts=time.time(),
            )
            self._persons[pid] = rec
            self._rebuild_matrix()
            self._dirty = True
            self.save()              
            logger.info("Đã thêm người mới ID=%d, tên=%s (K=%d)", pid, name, rec.count)
            return rec

    def rename_person(self, person_id: int, name: str) -> bool:
        name = " ".join(str(name).split())
        with self._lock:
            rec = self._persons.get(person_id)
            if rec is None or not name:
                return False
            rec.name = name
            rec.updated_at = _now_str()
            self._dirty = True
            self.save()
            return True

    def update_person(
        self,
        person_id: int,
        embedding: np.ndarray,
        min_interval: Optional[float] = None,
    ) -> bool:
        if min_interval is None:
            min_interval = config.FEATURE_UPDATE_MIN_INTERVAL
        emb = np.asarray(embedding, dtype=np.float32).reshape(-1)
        if emb.shape[0] != self.dim:
            return False
        now = time.time()
        with self._lock:
            rec = self._persons.get(person_id)
            if rec is None or (now - rec.last_update_ts) < min_interval:
                return False
            if rec.count >= self.max_k:
                rec.sum_vec = rec.sum_vec * (1.0 - 1.0 / self.max_k) + emb
            else:
                rec.sum_vec = rec.sum_vec + emb
            rec.count += 1
            rec.mean = l2_normalize(rec.sum_vec)
            rec.updated_at = _now_str()
            rec.last_update_ts = now
            idx = self._ids.index(person_id)
            self._matrix[idx] = rec.mean
            self._dirty = True
            return True
