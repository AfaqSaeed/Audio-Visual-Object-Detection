from __future__ import annotations

import csv
import hashlib
import argparse
import os
import shutil
import subprocess
import sys
import wave
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

# EPIC videos can trigger OpenCV's FFmpeg packet-read retry limit because they
# contain multiple streams. Raise the limit before importing cv2.
os.environ.setdefault("OPENCV_FFMPEG_READ_ATTEMPTS", "65536")

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = REPO_ROOT / "data"
DEFAULT_ANNOTATION_ROOT = REPO_ROOT / "epic-sounds-annotations"


@dataclass(frozen=True)
class SoundRow:
    annotation_id: str
    participant_id: str
    video_id: str
    start: float
    stop: float
    description: str
    class_name: str
    class_id: int


def parse_timestamp(ts: str) -> float:
    """
    Parse EPIC-style timestamps like "00:00:02.466" into seconds.
    """
    ts = ts.strip()
    # If your dataset uses comma decimals, normalize them.
    ts = ts.replace(",", ".")
    parts = ts.split(":")
    if len(parts) != 3:
        raise ValueError(f"Unexpected timestamp format: {ts}")
    h = int(parts[0])
    m = int(parts[1])
    s = float(parts[2])
    return h * 3600 + m * 60 + s


def class_id_to_color_bgr(class_id: int) -> tuple[int, int, int]:
    """
    Deterministic bright color for a given class_id (BGR).
    """
    # Use a stable hash -> hue.
    digest = hashlib.md5(str(class_id).encode("utf-8")).hexdigest()
    hue = int(digest[:8], 16) % 180  # OpenCV hue range for HSV is [0, 179]
    sat = 200
    val = 255

    hsv = np.uint8([[[hue, sat, val]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def load_sound_rows(csv_path: Path, video_id: str) -> list[SoundRow]:
    rows: list[SoundRow] = []
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if r.get("video_id") != video_id:
                continue

            start = parse_timestamp(r["start_timestamp"])
            stop = parse_timestamp(r["stop_timestamp"])
            class_id = int(r["class_id"])
            rows.append(
                SoundRow(
                    annotation_id=r.get("annotation_id", ""),
                    participant_id=r.get("participant_id", ""),
                    video_id=r.get("video_id", ""),
                    start=start,
                    stop=stop,
                    description=r.get("description", ""),
                    class_name=r.get("class", ""),
                    class_id=class_id,
                )
            )
    # Sort by start time for "most recently started" selection.
    rows.sort(key=lambda x: x.start)
    return rows


def find_first_video(video_root: Path) -> Path:
    exts = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    video_files = sorted([p for p in video_root.rglob("*") if p.is_file() and p.suffix.lower() in exts])
    if not video_files:
        raise FileNotFoundError(f"No video files found under: {video_root}")
    return video_files[0]


def find_video_by_id(video_root: Path, video_id: str) -> Path:
    exts = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    matches = sorted(
        [p for p in video_root.rglob("*") if p.is_file() and p.stem == video_id and p.suffix.lower() in exts]
    )
    if not matches:
        raise FileNotFoundError(f"Could not find video with id '{video_id}' under {video_root}")
    return matches[0]


def find_wav_for_video(sound_root: Path, video_id: str) -> Path | None:
    # Prefer exact match: <video_id>.wav
    for p in sound_root.rglob(f"{video_id}.wav"):
        if p.is_file():
            return p
    # Fallback: any wav whose stem equals the video_id.
    candidates = sorted([p for p in sound_root.rglob("*.wav") if p.is_file() and p.stem == video_id])
    return candidates[0] if candidates else None


def mux_audio_into_video(
    silent_video_path: Path,
    output_path: Path,
    audio_candidates: list[Path],
    duration_seconds: float,
) -> tuple[bool, str]:
    ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path:
        return False, "ffmpeg not found on PATH"

    for audio_source in audio_candidates:
        if not audio_source or not audio_source.exists():
            continue

        command = [
            ffmpeg_path,
            "-y",
            "-i",
            str(silent_video_path),
            "-i",
            str(audio_source),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-t",
            f"{max(0.0, duration_seconds):.3f}",
            "-shortest",
            str(output_path),
        ]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode == 0 and output_path.exists():
            return True, f"muxed audio from {audio_source}"

    return False, "no usable audio stream found in source video or wav"


def parse_ffprobe_key_value_output(text: str) -> dict[str, str]:
    data: dict[str, str] = {}
    for line in text.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        data[key.strip()] = value.strip()
    return data


def probe_video_with_ffmpeg(video_path: Path) -> tuple[float, int, int, int, float]:
    ffprobe_path = shutil.which("ffprobe")
    if not ffprobe_path:
        raise RuntimeError("ffprobe not found on PATH")

    command = [
        ffprobe_path,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,avg_frame_rate,r_frame_rate,nb_frames,duration",
        "-of",
        "default=noprint_wrappers=1",
        str(video_path),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {video_path}: {result.stderr.strip()}")

    meta = parse_ffprobe_key_value_output(result.stdout)
    width = int(meta["width"])
    height = int(meta["height"])

    fps_value = meta.get("avg_frame_rate") or meta.get("r_frame_rate") or "0/1"
    try:
        fps = float(Fraction(fps_value))
    except Exception as exc:
        raise RuntimeError(f"Could not parse fps '{fps_value}' for {video_path}") from exc
    if fps <= 0:
        raise RuntimeError(f"Invalid fps from ffprobe for {video_path}: {fps_value}")

    duration_value = meta.get("duration", "0")
    duration = float(duration_value) if duration_value not in {"N/A", ""} else 0.0

    nb_frames_value = meta.get("nb_frames", "0")
    if nb_frames_value not in {"N/A", ""}:
        frame_count = int(nb_frames_value)
    elif duration > 0:
        frame_count = int(round(duration * fps))
    else:
        frame_count = 0

    if duration <= 0 and frame_count > 0:
        duration = frame_count / fps

    return fps, frame_count, width, height, duration


class FfmpegVideoReader:
    def __init__(self, video_path: Path, width: int, height: int) -> None:
        ffmpeg_path = shutil.which("ffmpeg")
        if not ffmpeg_path:
            raise RuntimeError("ffmpeg not found on PATH")

        self.width = width
        self.height = height
        self.frame_bytes = width * height * 3
        command = [
            ffmpeg_path,
            "-v",
            "error",
            "-i",
            str(video_path),
            "-map",
            "0:v:0",
            "-an",
            "-sn",
            "-dn",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-vsync",
            "0",
            "-",
        ]
        self.process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self.process.stdout is None:
            return False, None

        raw = self.process.stdout.read(self.frame_bytes)
        if len(raw) != self.frame_bytes:
            return False, None

        # frombuffer() returns a read-only view over the ffmpeg stdout bytes.
        # Copy so OpenCV can draw overlays into the frame in-place.
        frame = np.frombuffer(raw, dtype=np.uint8).reshape((self.height, self.width, 3)).copy()
        return True, frame

    def release(self) -> None:
        if self.process.stdout is not None:
            self.process.stdout.close()
        if self.process.stderr is not None:
            self.process.stderr.close()
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)


def load_waveform_envelope(wav_path: Path | None, target_hz: int = 200) -> tuple[np.ndarray | None, float | None, str]:
    if wav_path is None or not wav_path.exists():
        return None, None, "waveform unavailable: wav not found"

    with wave.open(str(wav_path), "rb") as wf:
        num_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        sample_rate = wf.getframerate()
        num_frames = wf.getnframes()
        raw = wf.readframes(num_frames)

    if sample_width == 1:
        audio = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
        audio = (audio - 128.0) / 128.0
    elif sample_width == 2:
        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sample_width == 4:
        audio = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        return None, None, f"waveform unavailable: unsupported sample width {sample_width}"

    if num_channels > 1:
        audio = audio.reshape(-1, num_channels).mean(axis=1)

    audio = np.abs(audio)
    if audio.size == 0 or sample_rate <= 0:
        return None, None, "waveform unavailable: empty audio"

    samples_per_bin = max(1, int(sample_rate / target_hz))
    envelope_size = int(np.ceil(audio.size / samples_per_bin))
    pad = envelope_size * samples_per_bin - audio.size
    if pad > 0:
        audio = np.pad(audio, (0, pad), mode="constant")
    envelope = audio.reshape(envelope_size, samples_per_bin).max(axis=1)

    max_value = float(np.max(envelope))
    if max_value > 1e-6:
        envelope = envelope / max_value

    return envelope.astype(np.float32), float(target_hz), f"waveform loaded from {wav_path}"


def build_timeline_panel_canvas(
    rows: list[SoundRow],
    waveform_envelope: np.ndarray | None,
    waveform_hz: float | None,
    duration: float,
    panel_width: int,
    panel_height: int,
    timeline_window_seconds: float,
) -> tuple[np.ndarray, float]:
    pixels_per_second = max(1.0, panel_width / max(1e-6, timeline_window_seconds))
    canvas_width = max(panel_width + 2, int(np.ceil(duration * pixels_per_second)) + 2)
    canvas = np.full((panel_height, canvas_width, 3), 22, dtype=np.uint8)

    fill_y0 = int(0.14 * panel_height)
    fill_y1 = panel_height - int(0.14 * panel_height)
    center_y = int((fill_y0 + fill_y1) / 2)
    wave_amp = max(4, int((fill_y1 - fill_y0) * 0.45))
    seg_radius = max(5, int((fill_y1 - fill_y0) * 0.35))

    for r in rows:
        sx0 = int(r.start * pixels_per_second)
        sx1 = int(r.stop * pixels_per_second)
        sx0 = max(0, min(canvas_width - 1, sx0))
        sx1 = max(0, min(canvas_width - 1, sx1))
        if sx1 <= sx0:
            sx1 = min(canvas_width - 1, sx0 + 1)

        color = class_id_to_color_bgr(r.class_id)
        draw_rounded_rect(canvas, sx0, fill_y0, sx1, fill_y1, seg_radius, color, alpha=0.34)
        cv2.rectangle(canvas, (sx0, fill_y0), (sx1, fill_y1), color, 1, cv2.LINE_AA)

    if waveform_envelope is not None and waveform_hz is not None and waveform_envelope.size > 0:
        x_positions = np.arange(canvas_width, dtype=np.int32)
        times = x_positions / pixels_per_second
        envelope_indices = np.clip((times * waveform_hz).astype(np.int32), 0, waveform_envelope.size - 1)
        amplitudes = waveform_envelope[envelope_indices]

        points_top: list[tuple[int, int]] = []
        points_bottom: list[tuple[int, int]] = []
        for x, amp in zip(x_positions.tolist(), amplitudes.tolist()):
            amp_px = int(amp * wave_amp)
            points_top.append((x, center_y - amp_px))
            points_bottom.append((x, center_y + amp_px))

        polygon = np.array(points_top + points_bottom[::-1], dtype=np.int32)
        overlay = canvas.copy()
        cv2.fillPoly(overlay, [polygon], (230, 230, 230), lineType=cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.18, canvas, 0.82, 0, canvas)
        cv2.polylines(canvas, [np.array(points_top, dtype=np.int32)], False, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.polylines(canvas, [np.array(points_bottom, dtype=np.int32)], False, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.line(canvas, (0, center_y), (canvas_width - 1, center_y), (180, 180, 180), 1, cv2.LINE_AA)
    else:
        cv2.line(canvas, (0, center_y), (canvas_width - 1, center_y), (120, 120, 120), 1, cv2.LINE_AA)

    return canvas, pixels_per_second


def pick_active(rows: list[SoundRow], t: float) -> list[SoundRow]:
    # Assumes rows are sorted by start time. This is ok for moderate row counts.
    active: list[SoundRow] = []
    for r in rows:
        if r.start <= t <= r.stop:
            active.append(r)
        if r.start > t:
            break
    return active


def render_bottom_bar_and_text(
    frame: np.ndarray,
    t: float,
    duration: float,
    active_rows: list[SoundRow],
    rows: list[SoundRow],
    timeline_canvas: np.ndarray,
    timeline_pixels_per_second: float,
    waveform_envelope: np.ndarray | None,
    waveform_hz: float | None,
    bar_y0: int,
    bar_y1: int,
    timeline_window_seconds: float,
    font: int,
    font_scale: float,
    thickness: int,
    text_style: dict[str, int],
) -> None:
    if duration <= 0 or timeline_window_seconds <= 0:
        return

    h, w = frame.shape[:2]
    panel_margin_x = max(12, int(0.02 * w))
    panel_x0 = panel_margin_x
    panel_x1 = w - panel_margin_x
    panel_radius = max(8, int((bar_y1 - bar_y0) * 0.24))
    panel_color = (22, 22, 22)

    # Rounded container panel
    draw_rounded_rect(frame, panel_x0, bar_y0, panel_x1, bar_y1, panel_radius, panel_color, alpha=0.75)

    # Timeline window: [t - window, t]
    window_end = min(duration, max(0.0, t))
    window_start = max(0.0, window_end - timeline_window_seconds)
    window_span = max(1e-6, window_end - window_start)

    bar_height = max(1, bar_y1 - bar_y0)
    fill_y0 = bar_y0 + int(0.14 * bar_height)
    fill_y1 = bar_y1 - int(0.14 * bar_height)
    center_y = int((fill_y0 + fill_y1) / 2)

    # Subtle grid in the zoomed panel.
    n_grid = 8
    for i in range(n_grid + 1):
        gx = int(panel_x0 + (i / n_grid) * (panel_x1 - panel_x0))
        cv2.line(frame, (gx, bar_y0 + 2), (gx, bar_y1 - 2), (58, 58, 58), 1, cv2.LINE_AA)

    panel_w = max(1, panel_x1 - panel_x0)
    crop_x0 = max(0, int(window_start * timeline_pixels_per_second))
    crop_x1 = min(timeline_canvas.shape[1], max(crop_x0 + 2, int(np.ceil(window_end * timeline_pixels_per_second)) + 1))
    crop = timeline_canvas[:, crop_x0:crop_x1]
    if crop.size > 0:
        panel_img = cv2.resize(crop, (panel_w, bar_height), interpolation=cv2.INTER_LINEAR)
        roi = frame[bar_y0:bar_y1, panel_x0:panel_x1]
        if roi.shape[:2] == panel_img.shape[:2]:
            cv2.addWeighted(panel_img, 0.92, roi, 0.08, 0, roi)
    elif waveform_envelope is None or waveform_hz is None or waveform_envelope.size == 0:
        draw_centered_text_line(
            frame=frame,
            y_baseline=center_y + int(8 * font_scale),
            parts=[("Waveform unavailable", (220, 220, 220))],
            font=font,
            font_scale=max(0.55, font_scale * 0.8),
            thickness=max(1, thickness - 1),
            pad_x=text_style["pad_x"],
            pad_y=text_style["pad_y"],
            gap_px=text_style["gap_px"],
            bg_alpha=0.0,
            radius=text_style["radius"],
        )

    # Playhead at right edge (current time in lookback window).
    cv2.line(frame, (panel_x1 - 1, bar_y0), (panel_x1 - 1, bar_y1), (255, 255, 255), 2, cv2.LINE_AA)

    # Text overlay (above bar): centered + colored by class_id.
    active_class_ids = sorted({r.class_id for r in active_rows})
    y_text2 = bar_y0 - 10
    y_text1 = y_text2 - int(40 * font_scale)

    active_parts: list[tuple[str, tuple[int, int, int]]] = [("Active class_ids:", (235, 235, 235))]
    if active_class_ids:
        active_parts.append((" ", (235, 235, 235)))
        for idx, cid in enumerate(active_class_ids):
            token = str(cid) + ("," if idx < len(active_class_ids) - 1 else "")
            active_parts.append((token, class_id_to_color_bgr(cid)))
    else:
        active_parts.append((" (none)", (200, 200, 200)))

    draw_centered_text_line(
        frame=frame,
        y_baseline=y_text1,
        parts=active_parts,
        font=font,
        font_scale=font_scale,
        thickness=thickness,
        pad_x=text_style["pad_x"],
        pad_y=text_style["pad_y"],
        gap_px=text_style["gap_px"],
        bg_alpha=0.58,
        radius=text_style["radius"],
    )

    latest_desc = ""
    latest_id = None
    if active_rows:
        latest = max(active_rows, key=lambda r: r.start)
        latest_desc = latest.description
        latest_id = latest.class_id
        latest_class = latest.class_name

    if latest_desc and latest_class:
        text2 = f"Latest Description: {latest_desc} Latest Class: {latest_class} (class_id={latest_id})"
        latest_color = class_id_to_color_bgr(latest_id if latest_id is not None else 0)
    else:
        text2 = "Latest: (none)"
        latest_color = (220, 220, 220)
    draw_centered_text_line(
        frame=frame,
        y_baseline=y_text2,
        parts=[(text2, latest_color)],
        font=font,
        font_scale=font_scale,
        thickness=thickness,
        pad_x=text_style["pad_x"],
        pad_y=text_style["pad_y"],
        gap_px=text_style["gap_px"],
        bg_alpha=0.58,
        radius=text_style["radius"],
    )


def draw_rounded_rect(
    frame: np.ndarray,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    radius: int,
    color: tuple[int, int, int],
    alpha: float = 1.0,
) -> None:
    if x1 <= x0 or y1 <= y0:
        return
    radius = max(1, min(radius, (x1 - x0) // 2, (y1 - y0) // 2))
    overlay = frame.copy()

    cv2.rectangle(overlay, (x0 + radius, y0), (x1 - radius, y1), color, -1, cv2.LINE_AA)
    cv2.rectangle(overlay, (x0, y0 + radius), (x1, y1 - radius), color, -1, cv2.LINE_AA)
    cv2.circle(overlay, (x0 + radius, y0 + radius), radius, color, -1, cv2.LINE_AA)
    cv2.circle(overlay, (x1 - radius, y0 + radius), radius, color, -1, cv2.LINE_AA)
    cv2.circle(overlay, (x0 + radius, y1 - radius), radius, color, -1, cv2.LINE_AA)
    cv2.circle(overlay, (x1 - radius, y1 - radius), radius, color, -1, cv2.LINE_AA)

    if alpha >= 0.999:
        frame[:, :] = overlay
    else:
        cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)


def draw_centered_text_line(
    frame: np.ndarray,
    y_baseline: int,
    parts: list[tuple[str, tuple[int, int, int]]],
    font: int,
    font_scale: float,
    thickness: int,
    pad_x: int = 14,
    pad_y: int = 10,
    gap_px: int = 8,
    bg_color: tuple[int, int, int] = (0, 0, 0),
    bg_alpha: float = 0.58,
    radius: int = 14,
) -> None:
    h, w = frame.shape[:2]
    if not parts:
        return

    sizes: list[tuple[int, int]] = []
    total_w = 0
    max_h = 0
    for text, _ in parts:
        (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
        sizes.append((tw, th))
        total_w += tw
        max_h = max(max_h, th)
    total_w += gap_px * (len(parts) - 1)

    x0 = int((w - total_w) / 2)
    x0 = max(10, x0)
    x1 = min(w - 10, x0 + total_w)

    rect_x0 = max(8, x0 - pad_x)
    rect_x1 = min(w - 8, x1 + pad_x)
    rect_y0 = max(8, y_baseline - max_h - pad_y)
    rect_y1 = min(h - 8, y_baseline + pad_y)

    draw_rounded_rect(frame, rect_x0, rect_y0, rect_x1, rect_y1, radius, bg_color, alpha=bg_alpha)

    x = x0
    for (text, color), (tw, _th) in zip(parts, sizes):
        cv2.putText(frame, text, (x, y_baseline), font, font_scale, color, thickness, cv2.LINE_AA)
        x += tw + gap_px


def scaled_ui_value(base_value: int | float, resize_factor: float, minimum: int) -> int:
    return max(minimum, int(round(base_value * resize_factor)))


def scaled_text_style(resize_factor: float) -> dict[str, int]:
    ui_scale = max(0.35, resize_factor)
    return {
        "pad_x": scaled_ui_value(14, ui_scale, 6),
        "pad_y": scaled_ui_value(10, ui_scale, 4),
        "gap_px": scaled_ui_value(8, ui_scale, 3),
        "radius": scaled_ui_value(14, ui_scale, 5),
    }


def print_progress_bar(current: int, total: int, width: int = 32) -> None:
    if total <= 0:
        return

    current = max(0, min(current, total))
    ratio = current / total
    filled = int(ratio * width)
    bar = "#" * filled + "-" * (width - filled)
    message = f"\rRendering [{bar}] {current}/{total} ({ratio * 100:5.1f}%)"
    sys.stdout.write(message)
    sys.stdout.flush()
    if current >= total:
        sys.stdout.write("\n")
        sys.stdout.flush()


def main() -> None:
    parser = argparse.ArgumentParser(description="Overlay EPIC sound labels on a video.")
    parser.add_argument("--video_root", type=str, default=str(DEFAULT_DATA_ROOT / "EPIC-KITCHENS"))
    parser.add_argument("--sound_root", type=str, default=str(DEFAULT_DATA_ROOT / "sound-files"))
    parser.add_argument(
        "--train_csv",
        type=str,
        default=str(DEFAULT_ANNOTATION_ROOT / "EPIC_Sounds_train.csv"),
    )
    parser.add_argument(
        "--val_csv",
        type=str,
        default=str(DEFAULT_ANNOTATION_ROOT / "EPIC_Sounds_validation.csv"),
    )
    parser.add_argument("--video_id", type=str, default="")
    parser.add_argument(
        "--csv_split",
        type=str,
        choices=["train", "validation", "auto"],
        default="auto",
        help="Which CSV to use for rows (auto = prefer train if any rows exist).",
    )
    parser.add_argument(
        "--max_seconds",
        type=float,
        default=30.0,
        help="Render only the first N seconds for quick iteration (0 to render full video).",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=str(REPO_ROOT / "Visualize_Dataset" / "outputs"),
    )
    parser.add_argument(
        "--timeline_window_seconds",
        type=float,
        default=15.0,
        help="Lookback window size shown in zoomed timeline bar.",
    )
    parser.add_argument(
        "--resize_factor",
        type=float,
        default=1.0,
        help="Scale output frames by this factor for faster preview renders (e.g. 0.5).",
    )
    parser.add_argument(
        "--reader",
        type=str,
        choices=["auto", "opencv", "ffmpeg"],
        default="auto",
        help="Video decoding backend. 'auto' prefers ffmpeg when available.",
    )
    args = parser.parse_args()

    video_root = Path(args.video_root)
    sound_root = Path(args.sound_root)
    train_csv = Path(args.train_csv)
    val_csv = Path(args.val_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Pick one sample video (first found) and use its stem as video_id.
    if args.video_id:
        video_path = find_video_by_id(video_root, args.video_id)
        video_id = args.video_id
    else:
        video_path = find_first_video(video_root)
        video_id = video_path.stem
    wav_path = find_wav_for_video(sound_root, video_id)
    waveform_envelope, waveform_hz, waveform_status = load_waveform_envelope(wav_path)

    # Choose CSV (prefer train; fall back to val if no rows).
    train_rows = load_sound_rows(train_csv, video_id)
    val_rows = load_sound_rows(val_csv, video_id)
    if args.csv_split == "train":
        rows = train_rows
        source_csv = "train"
    elif args.csv_split == "validation":
        rows = val_rows
        source_csv = "validation"
    else:
        rows = train_rows if train_rows else val_rows
        source_csv = "train" if train_rows else "validation"

    reader_backend = args.reader
    ffmpeg_available = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None
    if reader_backend == "auto":
        reader_backend = "ffmpeg" if ffmpeg_available else "opencv"
    elif reader_backend == "ffmpeg" and not ffmpeg_available:
        raise RuntimeError("--reader ffmpeg requested, but ffmpeg/ffprobe are not available on PATH")

    cap = None
    ffmpeg_reader = None
    if reader_backend == "ffmpeg":
        fps, frame_count, width, height, duration = probe_video_with_ffmpeg(video_path)
        ffmpeg_reader = FfmpegVideoReader(video_path, width, height)
    else:
        cap = cv2.VideoCapture(str(video_path))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")
        if fps <= 0 or frame_count <= 0:
            cap.release()
            raise RuntimeError(f"Could not read fps/frame_count from: {video_path}")
        duration = frame_count / fps

    resize_factor = float(args.resize_factor)
    if resize_factor <= 0:
        if cap is not None:
            cap.release()
        if ffmpeg_reader is not None:
            ffmpeg_reader.release()
        raise ValueError("--resize_factor must be > 0")

    out_width = max(1, int(round(width * resize_factor)))
    out_height = max(1, int(round(height * resize_factor)))
    out_width += out_width % 2
    out_height += out_height % 2

    # Output paths
    out_path = out_dir / f"annotated_{video_id}.mp4"
    silent_out_path = out_dir / f"annotated_{video_id}.silent.mp4"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(silent_out_path), fourcc, fps, (out_width, out_height))

    font = cv2.FONT_HERSHEY_SIMPLEX
    ui_scale = max(0.35, resize_factor)
    font_scale = max(0.35, 0.75 * ui_scale)
    thickness = scaled_ui_value(2, ui_scale, 1)
    text_style = scaled_text_style(ui_scale)

    # Bottom regions bar dimensions
    bottom_bar_height = max(scaled_ui_value(70, ui_scale, 28), int(0.14 * out_height))
    bar_y1 = out_height - 1
    bar_y0 = out_height - bottom_bar_height
    panel_margin_x = max(12, int(0.02 * out_width))
    panel_width = max(1, (out_width - panel_margin_x) - panel_margin_x)
    panel_height = max(1, bar_y1 - bar_y0)
    timeline_canvas, timeline_pixels_per_second = build_timeline_panel_canvas(
        rows=rows,
        waveform_envelope=waveform_envelope,
        waveform_hz=waveform_hz,
        duration=duration,
        panel_width=panel_width,
        panel_height=panel_height,
        timeline_window_seconds=float(args.timeline_window_seconds),
    )

    # Frame loop
    frame_idx = 0
    max_seconds = float(args.max_seconds)
    if max_seconds <= 0:
        max_frames = frame_count
    else:
        max_frames = int(min(duration, max_seconds) * fps)
        max_frames = max(1, max_frames)

    print_progress_bar(0, max_frames)

    while True:
        if ffmpeg_reader is not None:
            ok, frame = ffmpeg_reader.read()
        else:
            ok, frame = cap.read()
        if not ok:
            break

        if resize_factor != 1.0:
            frame = cv2.resize(frame, (out_width, out_height), interpolation=cv2.INTER_AREA)

        t = frame_idx / fps
        active_rows = pick_active(rows, t)

        # Draw zoomed timeline + playhead + colored text.
        render_bottom_bar_and_text(
            frame=frame,
            t=t,
            duration=duration,
            active_rows=active_rows,
            rows=rows,
            timeline_canvas=timeline_canvas,
            timeline_pixels_per_second=timeline_pixels_per_second,
            waveform_envelope=waveform_envelope,
            waveform_hz=waveform_hz,
            bar_y0=bar_y0,
            bar_y1=bar_y1,
            timeline_window_seconds=float(args.timeline_window_seconds),
            font=font,
            font_scale=font_scale,
            thickness=thickness,
            text_style=text_style,
        )

        writer.write(frame)
        frame_idx += 1
        if frame_idx == 1 or frame_idx % 10 == 0 or frame_idx >= max_frames:
            print_progress_bar(frame_idx, max_frames)
        if frame_idx >= max_frames:
            break

    if cap is not None:
        cap.release()
    if ffmpeg_reader is not None:
        ffmpeg_reader.release()
    writer.release()

    rendered_duration = frame_idx / fps if fps > 0 else 0.0
    muxed_audio, mux_message = mux_audio_into_video(
        silent_video_path=silent_out_path,
        output_path=out_path,
        audio_candidates=[video_path, wav_path] if wav_path else [video_path],
        duration_seconds=rendered_duration,
    )
    if muxed_audio:
        silent_out_path.unlink(missing_ok=True)
    else:
        if out_path.exists():
            out_path.unlink()
        silent_out_path.replace(out_path)

    print(f"Video: {video_path}")
    print(f"video_id: {video_id}")
    print(f"wav: {wav_path if wav_path else 'NOT FOUND'}")
    print(f"Waveform: {waveform_status}")
    print(f"Loaded {len(rows)} rows from {source_csv} CSV")
    print(f"Resize factor: {resize_factor} -> output {out_width}x{out_height}")
    print(f"Reader: {reader_backend}")
    print(f"Audio: {mux_message}")
    print(f"Wrote annotated video: {out_path}")


if __name__ == "__main__":
    main()
