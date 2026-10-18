from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Set, Tuple

import numpy as np

import config
from utils.feature_manager import l2_normalize

STATUS_WAIT = "wait"        
STATUS_READY = "ready"      
STATUS_REJECT = "reject"    


@dataclass
class Candidate:
    cand_id: int
    sum_vec: np.ndarray
    first_ts: float
    last_ts: float
    last_sample_ts: float
    samples: List[np.ndarray] = field(default_factory=list)
    track_ids: Set[int] = field(default_factory=set)

    @property
    def centroid(self) -> np.ndarray:
        return l2_normalize(self.sum_vec)


class CandidatePool:
    def __init__(
        self,
        match_distance: float = None,
        min_samples: int = None,
        min_seconds: float = None,
        sample_interval: float = None,
        coherence_mean: float = None,
        coherence_min: float = None,
        ttl: float = None,
        max_candidates: int = None,
    ) -> None:
        self.match_distance = config.MATCH_DISTANCE_THRESHOLD if match_distance is None else match_distance
        self.min_samples = config.AUTO_ENROLL_MIN_SAMPLES if min_samples is None else min_samples
        self.min_seconds = config.AUTO_ENROLL_MIN_SECONDS if min_seconds is None else min_seconds
        self.sample_interval = config.AUTO_ENROLL_SAMPLE_INTERVAL if sample_interval is None else sample_interval
        self.coherence_mean = config.AUTO_ENROLL_COHERENCE_MEAN if coherence_mean is None else coherence_mean
        self.coherence_min = config.AUTO_ENROLL_COHERENCE_MIN if coherence_min is None else coherence_min
        self.ttl = config.AUTO_ENROLL_CANDIDATE_TTL if ttl is None else ttl
        self.max_candidates = config.AUTO_ENROLL_MAX_CANDIDATES if max_candidates is None else max_candidates
        self._cands: List[Candidate] = []
        self._next_id = 1
    def __len__(self) -> int:
        return len(self._cands)

    def reset(self) -> None:
        self._cands.clear()

    def observe(self, embedding: np.ndarray, track_id: int, now: float) -> Candidate:
        emb = np.asarray(embedding, dtype=np.float32).reshape(-1)
        best, best_d = None, self.match_distance
        for cand in self._cands:
            d = 1.0 - float(emb @ cand.centroid)
            if d < best_d:
                best, best_d = cand, d

        if best is None:
            if len(self._cands) >= self.max_candidates:        
                self._cands.remove(min(self._cands, key=lambda c: c.last_ts))
            best = Candidate(cand_id=self._next_id, sum_vec=emb.copy(), first_ts=now, last_ts=now,
                             last_sample_ts=now, samples=[emb.copy()])
            self._next_id += 1
            self._cands.append(best)
        elif now - best.last_sample_ts >= self.sample_interval:
            best.samples.append(emb.copy())
            best.sum_vec = best.sum_vec + emb
            best.last_sample_ts = now
        best.last_ts = now
        best.track_ids.add(track_id)
        return best

    def coherence(self, cand: Candidate) -> Tuple[float, float]:
        sims = np.stack(cand.samples) @ cand.centroid
        return float(sims.mean()), float(sims.min())

    def status(self, cand: Candidate) -> str:
        n = len(cand.samples)
        if n < self.min_samples:
            return STATUS_WAIT
        mean_sim, min_sim = self.coherence(cand)
        coherent = mean_sim >= self.coherence_mean and min_sim >= self.coherence_min
        if coherent and (cand.last_sample_ts - cand.first_ts) >= self.min_seconds:
            return STATUS_READY
        if not coherent and n >= 2 * self.min_samples:
            return STATUS_REJECT                     
        return STATUS_WAIT

    def remove(self, cand: Candidate) -> None:
        if cand in self._cands:
            self._cands.remove(cand)

    def expire(self, now: float) -> None:
        self._cands = [c for c in self._cands if now - c.last_ts <= self.ttl]
