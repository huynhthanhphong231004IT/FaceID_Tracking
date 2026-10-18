# -*- coding: utf-8 -*-
"""
main.py - Điểm khởi chạy ứng dụng Face Tracker (YOLO + ArcFace, giao diện CustomTkinter).

Cách chạy:
    python main.py            # tự dùng GPU nếu có
    python main.py --cpu      # ép chạy CPU
    python main.py --debug    # in log chi tiết
"""
from __future__ import annotations

import argparse
import importlib.util
import logging
import sys

import config

# (tên module để import, tên gói pip)
_REQUIRED = [
    ("numpy", "numpy"),
    ("cv2", "opencv-python"),
    ("PIL", "Pillow"),
    ("customtkinter", "customtkinter"),
    ("ultralytics", "ultralytics"),
    ("insightface", "insightface"),
    ("onnxruntime", "onnxruntime"),
]


def _check_dependencies() -> None:
    """Báo lỗi rõ ràng (thay vì traceback khó hiểu) khi thiếu thư viện."""
    missing = [pip for mod, pip in _REQUIRED if importlib.util.find_spec(mod) is None]
    if missing:
        print("\n[LỖI] Thiếu thư viện: " + ", ".join(missing))
        print("Hãy cài đặt bằng:  pip install -r requirements.txt\n")
        sys.exit(1)
    if importlib.util.find_spec("tkinter") is None:
        print("\n[LỖI] Python của bạn thiếu tkinter (cần cho giao diện).")
        print("Linux: sudo apt install python3-tk   |   Windows/macOS: cài lại Python từ python.org\n")
        sys.exit(1)


def main() -> int:
    parser = argparse.ArgumentParser(description="Real-time Multi-person Face Recognition & Re-ID")
    parser.add_argument("--cpu", action="store_true", help="Ép chạy bằng CPU (bỏ qua GPU)")
    parser.add_argument("--debug", action="store_true", help="Bật log chi tiết")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else getattr(logging, config.LOG_LEVEL, logging.INFO),
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    _check_dependencies()

    # Import muộn: chỉ sau khi chắc chắn đã đủ thư viện
    from gui.app_ui import FaceTrackerApp
    from pipeline import RecognitionPipeline

    pipeline = RecognitionPipeline(force_cpu=args.cpu)
    app = FaceTrackerApp(pipeline)
    app.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
