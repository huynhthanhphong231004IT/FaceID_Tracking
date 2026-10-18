
from __future__ import annotations

import logging
import os
import queue
import sys
import threading
import time
from typing import Optional, Union

import cv2
import numpy as np

import config

logger = logging.getLogger(__name__)


class VideoSourceError(RuntimeError):
    """Không mở/đọc được webcam hoặc file video."""


class VideoSource:
    def __init__(self, source: Union[int, str], is_file: bool = False,
                 queue_size: int = config.FRAME_QUEUE_SIZE) -> None:
        self.source = source
        self.is_file = is_file
        self._cap: Optional[cv2.VideoCapture] = None
        self._queue: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=max(1, queue_size))
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._ended = False

        self.error: Optional[str] = None
        self.fps: float = 30.0
        self.width: int = 0
        self.height: int = 0
        self.total_frames: int = 0

    def open(self) -> None:
        if self.is_file:
            path = str(self.source)
            if not os.path.isfile(path):
                raise VideoSourceError(f"Không tìm thấy file video:\n{path}")
            cap = cv2.VideoCapture(path)
        else:
            cap = None
            if sys.platform.startswith("win"):
                cap = cv2.VideoCapture(int(self.source), cv2.CAP_DSHOW)  # Mở nhanh hơn MSMF
                if not cap.isOpened():
                    cap.release()
                    cap = None
            if cap is None:
                cap = cv2.VideoCapture(int(self.source))
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)

        if not cap.isOpened():
            cap.release()
            if self.is_file:
                raise VideoSourceError(f"Không mở được video (file hỏng hoặc codec không hỗ trợ):\n{self.source}")
            raise VideoSourceError(
                f"Không mở được webcam (ID={self.source}).\n"
                "Hãy kiểm tra camera đã cắm/bật, chưa bị ứng dụng khác chiếm dụng, "
                "hoặc đổi CAMERA_ID trong config.py."
            )

        ok, probe = cap.read()
        if not ok or probe is None:
            cap.release()
            raise VideoSourceError("Mở được nguồn video nhưng không đọc được frame nào (file hỏng/camera lỗi).")
        if self.is_file:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)          

        fps = cap.get(cv2.CAP_PROP_FPS)
        self.fps = float(fps) if fps and 1.0 <= fps <= 240.0 else 30.0
        self.height, self.width = probe.shape[:2]
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) if self.is_file else 0
        self._cap = cap
        logger.info("Đã mở %s (%dx%d @ %.1f FPS)", self.source, self.width, self.height, self.fps)

    def start(self) -> None:
        if self._cap is None:
            raise VideoSourceError("Chưa gọi open().")
        self._thread = threading.Thread(target=self._run, daemon=True, name="capture")
        self._thread.start()

    def _run(self) -> None:
        cap = self._cap
        assert cap is not None
        interval = 1.0 / self.fps if self.is_file else 0.0
        next_t = time.perf_counter()
        failures = 0
        try:
            while not self._stop.is_set():
                ok, frame = cap.read()
                if not ok or frame is None:
                    if self.is_file:
                        break                                  # Hết video
                    failures += 1
                    if failures >= config.CAMERA_MAX_READ_FAILURES:
                        self.error = "Mất kết nối camera (không đọc được frame)."
                        break
                    time.sleep(0.03)
                    continue
                failures = 0

                if self.is_file:
                    next_t += interval                         
                    delay = next_t - time.perf_counter()
                    if delay > 0:
                        self._stop.wait(delay)
                    else:
                        next_t = time.perf_counter()
                    while not self._stop.is_set():             
                        try:
                            self._queue.put(frame, timeout=0.1)
                            break
                        except queue.Full:
                            continue
                else:
                    try:                                       
                        self._queue.put_nowait(frame)
                    except queue.Full:
                        try:
                            self._queue.get_nowait()
                        except queue.Empty:
                            pass
                        try:
                            self._queue.put_nowait(frame)
                        except queue.Full:
                            pass
        except Exception as exc: 
            logger.exception("Luồng đọc video gặp lỗi")
            self.error = f"Lỗi đọc video: {exc}"
        finally:
            self._ended = True
            cap.release()

    def read(self, timeout: float = 0.05) -> Optional[np.ndarray]:
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    @property
    def finished(self) -> bool:
        return self._ended and self._queue.empty()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None and self._thread.is_alive() and self._thread is not threading.current_thread():
            self._thread.join(timeout=2.0)
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception: 
                pass
        self._thread = None
