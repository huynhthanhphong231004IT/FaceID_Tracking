from __future__ import annotations
import functools
from typing import Optional, Tuple

import cv2
from PIL import Image, ImageDraw, ImageFont

import config
from utils.tracker import STATE_KNOWN, STATE_UNKNOWN

C = config.COLORS

_FONT_CANDIDATES = {
    False: ["arial.ttf", "segoeui.ttf", "DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/System/Library/Fonts/Supplemental/Arial.ttf", "/Library/Fonts/Arial.ttf"],
    True: ["arialbd.ttf", "segoeuib.ttf", "DejaVuSans-Bold.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
           "/System/Library/Fonts/Supplemental/Arial Bold.ttf", "/Library/Fonts/Arial Bold.ttf"],
}


@functools.lru_cache(maxsize=16)
def get_font(size: int, bold: bool = False):
    for name in _FONT_CANDIDATES[bold]:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:                       
        return ImageFont.load_default()


def state_color(state: str) -> str:
    if state == STATE_KNOWN:
        return C["lime"]
    if state == STATE_UNKNOWN:
        return C["magenta"]
    return C["amber"]


def face_label(face) -> str:
    if face.state == STATE_KNOWN:
        label = f"ID {face.person_id:05d}"
        if not face.name.startswith(config.AUTO_NAME_PREFIX):    
            label += f" · {face.name}"
        return f"{label} · {face.confidence * 100:.0f}%"
    if face.state == STATE_UNKNOWN:
        if face.learn:
            return f"UNKNOWN · LEARNING {face.learn[0]}/{face.learn[1]}"
        return "UNKNOWN"
    if face.is_good:
        return f"VERIFYING {face.vote_run}/{face.n_confirm}"
    return "FILTERED"


def _text_size(draw: ImageDraw.ImageDraw, text: str, font) -> Tuple[int, int]:
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    return right - left, bottom - top


def _draw_tag(draw, x: int, y: int, text: str, font, fg: str, bg: str, max_x: int) -> None:
    tw, th = _text_size(draw, text, font)
    pad = 4
    x = max(0, min(x, max_x - tw - 2 * pad))
    draw.rectangle([x, y, x + tw + 2 * pad, y + th + 2 * pad], fill=bg)
    draw.text((x + pad, y + pad - 1), text, font=font, fill=fg)


def _corners(draw, x1, y1, x2, y2, color: str, length: int, width: int) -> None:
    for sx, sy, dx, dy in ((x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)):
        draw.line([(sx, sy), (sx + dx * length, sy)], fill=color, width=width)
        draw.line([(sx, sy), (sx, sy + dy * length)], fill=color, width=width)


def render_frame(result, box_w: int, box_h: int, state_text: str = "", source_name: str = "",
                 banner: str = "") -> Tuple[Image.Image, float, float]:
    frame = result.frame
    h, w = frame.shape[:2]
    scale = min(max(box_w, 50) / w, max(box_h, 50) / h)
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    small = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    img = Image.fromarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    sx, sy = nw / w, nh / h

    font_main = get_font(max(12, min(18, nh // 38)), bold=True)
    font_small = get_font(max(10, min(14, nh // 52)))

    for face in result.faces:
        x1, y1, x2, y2 = (int(face.bbox[0] * sx), int(face.bbox[1] * sy),
                          int(face.bbox[2] * sx), int(face.bbox[3] * sy))
        color = state_color(face.state)
        selected = face.track_id == result.selected_track_id

        draw.rectangle([x1, y1, x2, y2], outline=color, width=1)
        corner = max(10, int(min(x2 - x1, y2 - y1) * 0.22))
        _corners(draw, x1, y1, x2, y2, color, corner, 3)
        if selected:                                           
            _corners(draw, x1 - 5, y1 - 5, x2 + 5, y2 + 5, C["cyan"], corner + 4, 2)

        text = face_label(face)
        _, th = _text_size(draw, text, font_main)
        ly = y1 - th - 12
        _draw_tag(draw, x1, ly if ly >= 0 else y1 + 2, text, font_main, "#04101c", color, nw)

        if face.yaw is not None:
            info = f"Y{face.yaw:+.0f}° P{face.pitch:+.0f}° R{face.roll:+.0f}°  " + ("OK" if face.is_good else "FILTERED")
        else:
            info = "NO POSE  FILTERED"
        _draw_tag(draw, x1, min(y2 + 3, nh - 22), info, font_small,
                  C["lime"] if face.is_good else C["red"], "#050912", nw)

    fps_text = f"FPS {result.fps:5.1f}   FACES {len(result.faces)}   DB {result.db_size}"
    _draw_tag(draw, 8, 8, fps_text, font_main, C["cyan"], "#050912", nw)
    if state_text or source_name:
        _draw_tag(draw, 8, nh - 28, f"{state_text}  {source_name}".strip(), font_small, C["muted"], "#050912", nw)
    if banner:
        font_big = get_font(max(20, nh // 14), bold=True)
        tw, th = _text_size(draw, banner, font_big)
        bx, by = (nw - tw) // 2, (nh - th) // 2
        draw.rectangle([bx - 18, by - 12, bx + tw + 18, by + th + 18], fill="#050912", outline=C["cyan"], width=2)
        draw.text((bx, by), banner, font=font_big, fill=C["cyan"])

    return img, sx, sy


def pick_face(result, x: float, y: float, sx: float, sy: float) -> Optional[int]:
    best, best_area = None, None
    for face in result.faces:
        x1, y1, x2, y2 = face.bbox
        if x1 * sx <= x <= x2 * sx and y1 * sy <= y <= y2 * sy:
            area = (x2 - x1) * (y2 - y1)
            if best is None or area < best_area:
                best, best_area = face.track_id, area
    return best
