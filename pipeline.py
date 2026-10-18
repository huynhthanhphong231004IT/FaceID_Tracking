from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

import config
from models.arcface_extractor import ArcFaceExtractor
from models.yolo_detector import YoloFaceDetector
from utils.candidate_pool import STATUS_READY, STATUS_REJECT, Candidate, CandidatePool
from utils.feature_manager import FeatureManager, l2_normalize
from utils.pose_estimator import PoseEstimator, QualityReport
from utils.tracker import (STATE_KNOWN, STATE_PENDING, STATE_UNKNOWN, FaceTracker, Track)
from utils.video_source import VideoSource

logger = logging.getLogger(__name__)

ST_IDLE, ST_RUNNING, ST_PAUSED, ST_FINISHED, ST_ERROR = "idle", "running", "paused", "finished", "error"
@dataclass
class FaceView:
    track_id: int
    bbox: Tuple[int, int, int, int]
    state: str                                  
    person_id: Optional[int]
    name: str
    confidence: float                          
    yaw: Optional[float]
    pitch: Optional[float]
    roll: Optional[float]
    sharpness: float
    is_good: bool                               
    reason: str                                 
    vote_run: int                                
    n_confirm: int
    feature_count: int                          
    crop: Optional[np.ndarray] = None           
    learn: Optional[Tuple[int, int]] = None     


@dataclass
class EnrollStatus:
    active: bool = False
    name: str = ""
    progress: int = 0
    total: int = config.ENROLL_SAMPLES
    message: str = ""


@dataclass
class FrameResult:
    seq: int
    frame: np.ndarray
    faces: List[FaceView]
    fps: float
    selected_track_id: Optional[int]
    enroll: EnrollStatus
    db_size: int
    timestamp: float = field(default_factory=time.time)


@dataclass
class _EnrollJob:
    name: str
    track_id: int
    samples: List[np.ndarray] = field(default_factory=list)
    started: float = field(default_factory=time.time)
    last_sample_ts: float = 0.0
    lost_frames: int = 0

class RecognitionPipeline:
    def __init__(self, force_cpu: bool = False) -> None:
        self.force_cpu = force_cpu
        self.db = FeatureManager()
        self.pose = PoseEstimator()
        self.extractor = ArcFaceExtractor(force_cpu=force_cpu)
        self.detector = YoloFaceDetector(landmark_fallback=self.extractor.landmarks_from_crop,
                                         force_cpu=force_cpu)
        self.models_ready = False
        self.events: "queue.Queue[Tuple[str, str]]" = queue.Queue(maxsize=500)   # (mức, nội dung) cho GUI
        self.source_name = ""
        self._lock = threading.RLock()
        self._latest: Optional[FrameResult] = None
        self._seq = 0
        self._capture: Optional[VideoSource] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_evt = threading.Event()
        self._paused = threading.Event()
        self._state = ST_IDLE
        self._selected_track_id: Optional[int] = None
        self._enroll: Optional[_EnrollJob] = None
        self._enroll_message = ""
        self._enroll_message_ts = 0.0
        self._pool = CandidatePool()                     

    def load_models(self) -> None:
        self.db.load()
        self.extractor.load()       
        self.detector.load()
        self.models_ready = True
        self._emit("ok", f"Model sẵn sàng. DB hiện có {self.db.num_persons()} người.")
    @property
    def state(self) -> str:
        return self._state

    def start(self, source: Union[int, str], is_file: bool = False, label: str = "") -> None:
        if not self.models_ready:
            raise RuntimeError("Model chưa sẵn sàng.")
        self.stop()

        capture = VideoSource(source, is_file=is_file)
        capture.open()                                   
        capture.start()

        with self._lock:
            self._capture = capture
            self._latest = None
            self._selected_track_id = None
            self._enroll = None
            self._pool = CandidatePool()                 
            self._stop_evt = threading.Event()
            self._paused.clear()
            self.source_name = label or str(source)
            self._state = ST_RUNNING
            self._thread = threading.Thread(
                target=self._inference_loop,
                args=(capture, FaceTracker(), self._stop_evt),
                daemon=True, name="inference",
            )
            self._thread.start()
        self._emit("info", f"Bắt đầu: {self.source_name} ({capture.width}x{capture.height} @ {capture.fps:.0f} FPS)")

    def stop(self) -> None:
        with self._lock:
            self._stop_evt.set()
            thread, capture = self._thread, self._capture
        if thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=3.0)
        if capture is not None:
            capture.stop()
        with self._lock:
            self._thread = None
            self._capture = None
            self._latest = None
            self._enroll = None
            self._selected_track_id = None
            self._paused.clear()
            self._state = ST_IDLE
        self.db.save(force=True)

    def pause(self) -> None:
        if self._state == ST_RUNNING:
            self._paused.set()
            self._state = ST_PAUSED

    def resume(self) -> None:
        if self._state == ST_PAUSED:
            self._paused.clear()
            self._state = ST_RUNNING

    def toggle_pause(self) -> bool:
        if self._state == ST_PAUSED:
            self.resume()
        else:
            self.pause()
        return self._state == ST_PAUSED

    def shutdown(self) -> None:
        self.stop()
        self.db.save(force=True)

    def get_latest(self) -> Optional[FrameResult]:
        with self._lock:
            return self._latest

    def select_track(self, track_id: Optional[int]) -> None:
        with self._lock:
            self._selected_track_id = track_id
            latest = self._latest
            if latest is not None and track_id is not None and any(f.track_id == track_id for f in latest.faces):
                latest.selected_track_id = track_id

    def begin_enrollment(self, name: str) -> Tuple[bool, str]:
        with self._lock:
            latest = self._latest
            if self._state not in (ST_RUNNING, ST_PAUSED) or latest is None or not latest.faces:
                return False, "Không thấy khuôn mặt nào trong khung hình để đăng ký."
            if self._enroll is not None:
                return False, f"Đang đăng ký '{self._enroll.name}', hãy đợi hoặc huỷ trước."

            target = next((f for f in latest.faces if f.track_id == latest.selected_track_id), None)
            if target is None:
                target = max(latest.faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
            if target.state == STATE_KNOWN:
                if self.db.is_auto_name(target.name) and self.db.rename_person(target.person_id, name):
                    msg = f"Đã đặt tên '{name}' cho ID {target.person_id:05d}."
                    self._set_enroll_message(msg)
                    self._emit("ok", msg)
                    return True, msg
                return False, (f"Khuôn mặt đang chọn đã được nhận diện là '{target.name}' "
                               f"(ID {target.person_id:05d}). Hãy chọn một khuôn mặt 'Unknown'.")
            self._enroll = _EnrollJob(name=name, track_id=target.track_id)
            self._selected_track_id = target.track_id
        msg = (f"Đang thu thập {config.ENROLL_SAMPLES} frame chuẩn cho '{name}'. "
               "Hãy nhìn thẳng camera và xoay mặt nhẹ.")
        self._emit("info", msg)
        return True, msg

    def cancel_enrollment(self, message: str = "Đã huỷ đăng ký.") -> None:
        with self._lock:
            had_job = self._enroll is not None
            self._enroll = None
        if had_job:
            self._set_enroll_message(message)
            self._emit("warn", message)

    def _emit(self, level: str, text: str) -> None:
        try:
            self.events.put_nowait((level, text))
        except queue.Full:
            pass

    def _set_enroll_message(self, text: str) -> None:
        self._enroll_message = text
        self._enroll_message_ts = time.time()

    def _inference_loop(self, capture: VideoSource, tracker: FaceTracker, stop_evt: threading.Event) -> None:
        fps = 0.0
        last_t = time.perf_counter()
        consecutive_errors = 0
        last_error_text = ""

        while not stop_evt.is_set():
            if self._paused.is_set():
                time.sleep(0.03)
                last_t = time.perf_counter()
                continue

            frame = capture.read(timeout=0.05)
            if frame is None:
                if capture.error:
                    self._emit("error", capture.error)
                    self._state = ST_ERROR
                    break
                if capture.finished:
                    self._emit("info", "Video đã phát hết.")
                    self._state = ST_FINISHED
                    break
                continue

            try:
                faces, selected, enroll_status = self._process_frame(frame, tracker)
                consecutive_errors = 0
            except Exception as exc:  
                consecutive_errors += 1
                text = f"Lỗi xử lý frame: {exc}"
                logger.exception("Lỗi xử lý frame")
                if text != last_error_text:
                    self._emit("error", text)
                    last_error_text = text
                if consecutive_errors >= 15:
                    self._emit("error", "Quá nhiều lỗi liên tiếp, dừng xử lý.")
                    self._state = ST_ERROR
                    break
                continue

            now = time.perf_counter()
            inst = 1.0 / max(now - last_t, 1e-6)
            fps = inst if fps == 0.0 else 0.9 * fps + 0.1 * inst
            last_t = now

            self._seq += 1
            result = FrameResult(seq=self._seq, frame=frame, faces=faces, fps=fps,
                                 selected_track_id=selected, enroll=enroll_status,
                                 db_size=self.db.num_persons())
            with self._lock:
                self._latest = result
            self.db.autosave()

    def _process_frame(
        self, frame: np.ndarray, tracker: FaceTracker
    ) -> Tuple[List[FaceView], Optional[int], EnrollStatus]:
        detections = self.detector.detect(frame)
        tracks = tracker.update([d.bbox for d in detections])
        good_idx: List[int] = []
        aligned_faces: List[np.ndarray] = []
        for i, (det, track) in enumerate(zip(detections, tracks)):
            if det.kps is None:
                report = QualityReport(reason="Không xác định được 5 landmark")
                aligned = None
            else:
                aligned = self.extractor.align(frame, det.kps)
                report = self.pose.assess(det.bbox, det.kps, aligned, frame.shape)
            track.last_quality = report
            if report.is_good and aligned is not None:
                good_idx.append(i)
                aligned_faces.append(aligned)
        embeddings = self.extractor.embed(aligned_faces)
        matches = self._match(embeddings)        
        now = time.time()
        learn: Dict[int, Tuple[int, int]] = {}
        with self._lock:
            manual_target = self._enroll.track_id if self._enroll is not None else None
        for local, det_i in enumerate(good_idx):
            track = tracks[det_i]
            person_id, similarity, nearest = matches[local]
            before = (track.state, track.person_id)
            changed = track.add_vote(person_id, similarity)
            if changed and (track.state, track.person_id) != before:
                self._log_transition(track)
            if (track.state == STATE_KNOWN and person_id is not None and person_id == track.person_id
                    and (1.0 - similarity) <= config.UPDATE_DISTANCE_THRESHOLD):
                self.db.update_person(person_id, embeddings[local])
            if (config.AUTO_ENROLL_ENABLED and track.state == STATE_UNKNOWN and person_id is None
                    and track.track_id != manual_target
                    and min(detections[det_i].bbox[2] - detections[det_i].bbox[0],
                            detections[det_i].bbox[3] - detections[det_i].bbox[1]) >= config.AUTO_ENROLL_MIN_FACE_SIZE
                    and nearest >= config.AUTO_ENROLL_MIN_DB_DISTANCE):
                cand = self._pool.observe(embeddings[local], track.track_id, now)
                learn[track.track_id] = (len(cand.samples), config.AUTO_ENROLL_MIN_SAMPLES)
                status = self._pool.status(cand)
                if status == STATUS_READY:
                    self._auto_enroll(cand, tracks)
                elif status == STATUS_REJECT:
                    self._pool.remove(cand)
                    self._emit("warn", "Bỏ một cụm khuôn mặt lạ không đồng nhất (không tạo ID).")
        self._pool.expire(now)

        enroll_status = self._enrollment_step(tracks, good_idx, embeddings, tracker)
        faces: List[FaceView] = []
        for det, track in zip(detections, tracks):
            rep = track.last_quality or QualityReport()
            known = track.state == STATE_KNOWN
            faces.append(FaceView(
                track_id=track.track_id, bbox=det.bbox, state=track.state,
                person_id=track.person_id if known else None,
                name=self.db.get_name(track.person_id) if known else "Unknown",
                confidence=track.similarity if known else 0.0,
                yaw=rep.yaw, pitch=rep.pitch, roll=rep.roll, sharpness=rep.sharpness,
                is_good=rep.is_good, reason=rep.reason,
                vote_run=track.vote_run(), n_confirm=track.n_confirm,
                feature_count=self.db.get_count(track.person_id) if known else 0,
                crop=self._crop(frame, det.bbox),
                learn=learn.get(track.track_id) if track.state == STATE_UNKNOWN else None,
            ))

        selected = self._resolve_selected(faces)
        return faces, selected, enroll_status

    def _match(self, embeddings: np.ndarray) -> List[Tuple[Optional[int], float, float]]:
        n = embeddings.shape[0]
        if n == 0:
            return []
        dist, ids = self.db.distance_matrix(embeddings)
        if not ids:
            return [(None, 0.0, 1.0)] * n              

        nearest = dist.min(axis=1)
        pairs = sorted(
            (float(dist[r, c]), r, c) for r in range(n) for c in range(len(ids))
            if dist[r, c] < config.MATCH_DISTANCE_THRESHOLD
        )
        out: List[Tuple[Optional[int], float, float]] = [(None, 0.0, float(nearest[r])) for r in range(n)]
        used_r, used_c = set(), set()
        for d, r, c in pairs:
            if r in used_r or c in used_c:
                continue
            used_r.add(r)
            used_c.add(c)
            out[r] = (ids[c], 1.0 - d, float(nearest[r]))
        return out

    def _auto_enroll(self, cand: Candidate, tracks: List[Track]) -> None:
        centroid = cand.centroid
        dist, ids = self.db.distance_matrix(centroid[None, :])
        if ids and float(dist.min()) < config.AUTO_ENROLL_MIN_DB_DISTANCE:
            self._pool.remove(cand)                    
            return
        record = self.db.add_person(None, cand.samples)
        self._pool.remove(cand)
        for track in tracks:                             
            if track.track_id in cand.track_ids and track.state == STATE_UNKNOWN:
                track.set_identity(record.person_id, similarity=float(cand.centroid @ centroid))
        self._emit("ok", f"Tự đăng ký người lạ → ID {record.person_id:05d} "
                         f"({record.count} mẫu, quan sát {cand.last_sample_ts - cand.first_ts:.1f}s)")

    def _log_transition(self, track: Track) -> None:
        if track.state == STATE_KNOWN:
            self._emit("ok", f"Track #{track.track_id} xác nhận: ID {track.person_id:05d} · "
                             f"{self.db.get_name(track.person_id)} (sau {track.n_confirm} frame chuẩn)")
        elif track.state == STATE_UNKNOWN:
            self._emit("warn", f"Track #{track.track_id}: khuôn mặt lạ (không có trong DB)")

    @staticmethod
    def _crop(frame: np.ndarray, bbox: Tuple[int, int, int, int], margin: float = 0.2,
              max_side: int = 224) -> Optional[np.ndarray]:
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        mx, my = int((x2 - x1) * margin), int((y2 - y1) * margin)
        crop = frame[max(0, y1 - my):min(h, y2 + my), max(0, x1 - mx):min(w, x2 + mx)]
        if crop.size == 0:
            return None
        scale = max(crop.shape[:2]) / float(max_side)
        if scale > 1.0:
            crop = cv2.resize(crop, (int(crop.shape[1] / scale), int(crop.shape[0] / scale)),
                              interpolation=cv2.INTER_AREA)
        return crop.copy()

    def _resolve_selected(self, faces: List[FaceView]) -> Optional[int]:
        if not faces:
            return None
        with self._lock:
            wanted = self._selected_track_id
        if wanted is not None and any(f.track_id == wanted for f in faces):
            return wanted
        biggest = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        return biggest.track_id

    def _enrollment_step(self, tracks: List[Track], good_idx: List[int],
                         embeddings: np.ndarray, tracker: FaceTracker) -> EnrollStatus:
        with self._lock:
            job = self._enroll
        if job is None:
            recent = (time.time() - self._enroll_message_ts) < 6.0
            return EnrollStatus(active=False, message=self._enroll_message if recent else "")

        now = time.time()
        total = config.ENROLL_SAMPLES
        position = next((i for i, t in enumerate(tracks) if t.track_id == job.track_id), None)

        if position is None:
            job.lost_frames += 1
            if job.lost_frames > 30:
                self.cancel_enrollment("Mất dấu khuôn mặt đang đăng ký — đã huỷ.")
                return EnrollStatus(active=False, message=self._enroll_message)
        else:
            job.lost_frames = 0
            if position in good_idx and (now - job.last_sample_ts) >= config.ENROLL_MIN_INTERVAL:
                job.samples.append(embeddings[good_idx.index(position)].copy())
                job.last_sample_ts = now

        if now - job.started > config.ENROLL_TIMEOUT_SEC and len(job.samples) < total:
            self.cancel_enrollment("Hết thời gian: không đủ frame chuẩn. Hãy nhìn thẳng camera và thử lại.")
            return EnrollStatus(active=False, message=self._enroll_message)

        if len(job.samples) >= total:
            self._finish_enrollment(job, tracks[position] if position is not None else None)
            with self._lock:
                self._enroll = None
            return EnrollStatus(active=False, message=self._enroll_message)

        return EnrollStatus(active=True, name=job.name, progress=len(job.samples), total=total,
                            message=f"Thu thập mẫu cho '{job.name}'...")

    def _finish_enrollment(self, job: _EnrollJob, track: Optional[Track]) -> None:
        mean = l2_normalize(np.mean(np.stack(job.samples), axis=0))
        if config.ENROLL_REJECT_DUPLICATE:
            dist, ids = self.db.distance_matrix(mean[None, :])
            if ids and float(dist.min()) < config.MATCH_DISTANCE_THRESHOLD:
                existing = ids[int(dist.argmin())]
                msg = (f"Từ chối: khuôn mặt này giống '{self.db.get_name(existing)}' (ID {existing}) "
                       f"đã có trong DB (khoảng cách {float(dist.min()):.2f}).")
                self._set_enroll_message(msg)
                self._emit("warn", msg)
                return
        record = self.db.add_person(job.name, job.samples)
        if track is not None:
            track.set_identity(record.person_id, similarity=1.0)
        msg = f"Đã thêm '{record.name}' với ID {record.person_id} ({record.count} đặc trưng)."
        self._set_enroll_message(msg)
        self._emit("ok", msg)
